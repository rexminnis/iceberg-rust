// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashSet;

use crate::error::Result;
use crate::spec::{DataFile, ManifestFile, ManifestStatus};
use crate::table::Table;
use crate::transaction::snapshot::SnapshotProducer;
use crate::{Error, ErrorKind};

/// Accumulates the set of files an operation removes from a table and rewrites the
/// affected manifests during manifest production.
///
/// A delete-class operation owns one `ManifestFilterManager` for data manifests and one for
/// delete manifests. Removed files are recorded via [`ManifestFilterManager::delete_file`]
/// keyed by their file path; the later filtering pass drops the matching manifest entries
/// and re-emits the survivors.
///
/// **In v1 this always rewrites**: there is no cross-retry rewrite cache (that is deferred
/// to future work). The single `Mutex<MergingCache>` on `MergingSnapshotProducer` is
/// retained precisely so the deferred cache has a home.
#[derive(Default)]
pub(crate) struct ManifestFilterManager {
    /// Files to drop, keyed by file path. `DataFile` covers both data and delete files.
    deleted_files: HashSet<String>,
}

impl ManifestFilterManager {
    /// Record a file for removal.
    ///
    /// `DataFile` covers both data and delete files, so the same entry point serves
    /// data-manifest and delete-manifest filtering. Removals are keyed by file path;
    /// recording the same path more than once keeps a single removal entry.
    pub(crate) fn delete_file(&mut self, file: DataFile) {
        self.deleted_files.insert(file.file_path().to_string());
    }

    /// Returns `true` if the file at `path` is recorded for removal.
    pub(crate) fn is_removed(&self, path: &str) -> bool {
        self.deleted_files.contains(path)
    }

    /// Rewrite the given `manifests`, dropping any entries recorded for removal and
    /// re-emitting the survivors via a producer-provided manifest writer.
    ///
    /// Manifests that contain none of the removed files are passed through unchanged, so
    /// the common (untouched) case costs one manifest read and no write. For each manifest
    /// that does reference a removed file, a new manifest is written that:
    ///
    /// - skips entries already marked [`ManifestStatus::Deleted`] (informational only,
    ///   never carried forward), and
    /// - skips entries whose file path is in the removed set.
    ///
    /// All remaining live entries are re-emitted as existing entries, preserving their
    /// original snapshot id and sequence numbers. If a rewrite drops every entry, the
    /// manifest is omitted from the returned vector entirely.
    ///
    /// `fail_missing_delete_paths` mirrors Java's `failMissingDeletePaths`: when `true`,
    /// every recorded removal must be found among the live entries of the given manifests,
    /// otherwise the commit fails with `PreconditionFailed`. A rewrite (compaction) sets
    /// this — committing the *added* half of a rewrite while the *remove* half silently
    /// no-ops would duplicate every rewritten row.
    ///
    /// **v1 always rewrites**: every affected manifest goes through the rewrite path
    /// unconditionally on every attempt — there is no cross-retry cache (deferred; would
    /// live in `MergingCache`).
    pub(crate) async fn filter_manifests(
        &self,
        sp: &mut SnapshotProducer<'_>,
        base: &Table,
        manifests: Vec<ManifestFile>,
        fail_missing_delete_paths: bool,
    ) -> Result<Vec<ManifestFile>> {
        // Nothing recorded for removal: every manifest is carried forward verbatim.
        if self.deleted_files.is_empty() {
            return Ok(manifests);
        }

        // Paths this operation wants to remove but has not yet seen among live entries.
        // Whatever remains after the loop is missing from the table; with
        // `fail_missing_delete_paths` that is a commit-stopping error.
        let mut pending_deletes: HashSet<String> = self.deleted_files.clone();

        let file_io = base.file_io().clone();
        let mut filtered = Vec::with_capacity(manifests.len());

        for manifest_file in manifests {
            let manifest = manifest_file.load_manifest(&file_io).await?;
            let entries = manifest.entries();

            // Pass the manifest through unchanged when none of its live entries reference
            // a removed file. Deleted entries are ignored for this decision.
            let has_removed_entry = entries.iter().any(|entry| {
                entry.status() != ManifestStatus::Deleted && self.is_removed(entry.file_path())
            });
            if !has_removed_entry {
                filtered.push(manifest_file);
                continue;
            }

            // Rewrite the manifest, re-emitting the survivors as existing entries.
            // The filtered manifest is written with the SOURCE manifest's partition
            // spec — survivors' partition values were shaped by it, and a table that
            // has evolved its spec still carries old-spec manifests (Java behaves the
            // same way: filtered manifests keep their spec).
            let mut writer = sp.new_manifest_writer_with_spec(
                manifest_file.content,
                manifest_file.partition_spec_id,
            )?;
            let mut survivors = 0usize;
            for entry in entries {
                // Deleted entries are informational only; never carried forward.
                if entry.status() == ManifestStatus::Deleted {
                    continue;
                }

                if self.is_removed(entry.file_path()) {
                    // Drop the entry: this is the removal actually happening.
                    pending_deletes.remove(entry.file_path());
                    continue;
                }

                writer.add_existing_file(
                    entry.data_file().clone(),
                    entry.snapshot_id().unwrap_or_default(),
                    entry.sequence_number().unwrap_or_default(),
                    entry.file_sequence_number,
                )?;
                survivors += 1;
            }

            // If every entry was dropped, omit the manifest from the output entirely.
            if survivors == 0 {
                continue;
            }

            filtered.push(writer.write_manifest_file().await?);
        }

        if fail_missing_delete_paths && !pending_deletes.is_empty() {
            let mut missing: Vec<&String> = pending_deletes.iter().collect();
            missing.sort();
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                format!("Missing required files to delete: {missing:?}"),
            ));
        }

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;

    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::TableIdent;
    use crate::io::FileIO;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry, ManifestStatus,
        ManifestWriterBuilder, Struct, TableMetadata,
    };
    use crate::table::Table;
    use crate::test_utils::test_runtime;
    use crate::transaction::snapshot::SnapshotProducer;

    fn data_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// Builds a v2 table backed by the local filesystem in a `TempDir`, so manifests can
    /// be written to and loaded from real files (as `filter_manifests` requires).
    fn make_fs_table() -> (Table, TempDir) {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().join("table1");
        let manifest_list_location = table_location.join("metadata/manifests_list_1.avro");
        let table_metadata_location = table_location.join("metadata/v1.json");

        let file_io = FileIO::new_with_fs();

        let template = fs::read_to_string(format!(
            "{}/testdata/example_table_metadata_v2.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let metadata_json = template
            .replace("{{ table_location }}", table_location.to_str().unwrap())
            .replace(
                "{{ manifest_list_1_location }}",
                manifest_list_location.to_str().unwrap(),
            )
            .replace(
                "{{ manifest_list_2_location }}",
                manifest_list_location.to_str().unwrap(),
            )
            .replace(
                "{{ table_metadata_1_location }}",
                table_metadata_location.to_str().unwrap(),
            );
        let table_metadata = serde_json::from_str::<TableMetadata>(&metadata_json).unwrap();

        let table = Table::builder()
            .metadata(table_metadata)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(file_io)
            .metadata_location(table_metadata_location.to_str().unwrap())
            .runtime(test_runtime())
            .build()
            .unwrap();

        (table, tmp_dir)
    }

    fn make_producer<'a>(table: &'a Table) -> SnapshotProducer<'a> {
        SnapshotProducer::builder()
            .table(table)
            .commit_uuid(Uuid::now_v7())
            .snapshot_properties(HashMap::new())
            .added_data_files(vec![])
            .build()
    }

    /// Writes a data manifest to disk containing one `Added` data entry per path in
    /// `paths` and returns the resulting [`ManifestFile`].
    async fn write_data_manifest(table: &Table, paths: &[&str]) -> ManifestFile {
        let current_snapshot = table.metadata().current_snapshot().unwrap();
        let schema = current_snapshot.schema(table.metadata()).unwrap();
        let partition_spec = table.metadata().default_partition_spec();
        let table_location_str = table.metadata().location().to_string();

        let output_file = table
            .file_io()
            .new_output(format!(
                "{table_location_str}/metadata/manifest_{}.avro",
                Uuid::new_v4()
            ))
            .unwrap();

        let mut writer = ManifestWriterBuilder::new(
            output_file,
            Some(current_snapshot.snapshot_id()),
            schema.clone(),
            partition_spec.as_ref().clone(),
        )
        .build_v2_data();

        for path in paths {
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(path.to_string())
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(100)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(100))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
        }

        writer.write_manifest_file().await.unwrap()
    }

    /// Collects the file paths of all alive entries in a manifest on disk.
    async fn manifest_file_paths(table: &Table, manifest_file: &ManifestFile) -> Vec<String> {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        manifest
            .entries()
            .iter()
            .filter(|entry| entry.status() != ManifestStatus::Deleted)
            .map(|entry| entry.file_path().to_string())
            .collect()
    }

    #[test]
    fn test_delete_file_records_removal_by_path_and_dedupes() {
        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("test/1.parquet"));
        manager.delete_file(data_file("test/1.parquet"));

        assert!(manager.is_removed("test/1.parquet"));
        assert!(!manager.is_removed("test/2.parquet"));
        assert_eq!(manager.deleted_files.len(), 1);
    }

    #[tokio::test]
    async fn test_filter_manifests_no_op_when_nothing_removed() {
        let (table, _tmp_dir) = make_fs_table();
        let manifest = write_data_manifest(&table, &["data/a.parquet", "data/b.parquet"]).await;
        let mut producer = make_producer(&table);

        // Nothing recorded for removal: the manifest must be returned verbatim.
        let manager = ManifestFilterManager::default();
        let result = manager
            .filter_manifests(&mut producer, &table, vec![manifest.clone()], true)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].manifest_path, manifest.manifest_path);
    }

    #[tokio::test]
    async fn test_filter_manifests_untouched_manifest_passes_through_unchanged() {
        let (table, _tmp_dir) = make_fs_table();
        let touched = write_data_manifest(&table, &["data/a.parquet", "data/b.parquet"]).await;
        let untouched = write_data_manifest(&table, &["data/x.parquet"]).await;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/a.parquet"));

        let result = manager
            .filter_manifests(
                &mut producer,
                &table,
                vec![touched.clone(), untouched.clone()],
                true,
            )
            .await
            .unwrap();

        // The untouched manifest is carried forward verbatim (same path — no rewrite);
        // the touched one is rewritten to a new path.
        assert_eq!(result.len(), 2);
        assert_ne!(result[0].manifest_path, touched.manifest_path);
        assert_eq!(result[1].manifest_path, untouched.manifest_path);
    }

    #[tokio::test]
    async fn test_filter_manifests_removing_all_entries_drops_manifest() {
        let (table, _tmp_dir) = make_fs_table();
        let manifest = write_data_manifest(&table, &["data/a.parquet", "data/b.parquet"]).await;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/a.parquet"));
        manager.delete_file(data_file("data/b.parquet"));

        let result = manager
            .filter_manifests(&mut producer, &table, vec![manifest], true)
            .await
            .unwrap();

        // The manifest is dropped from the output entirely once all entries are removed.
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_filter_manifests_removing_subset_rewrites_survivors_as_existing() {
        let (table, _tmp_dir) = make_fs_table();
        let original = write_data_manifest(&table, &[
            "data/a.parquet",
            "data/b.parquet",
            "data/c.parquet",
        ])
        .await;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/b.parquet"));

        let result = manager
            .filter_manifests(&mut producer, &table, vec![original.clone()], true)
            .await
            .unwrap();

        // A rewritten manifest is produced (distinct from the original) holding survivors.
        assert_eq!(result.len(), 1);
        assert_ne!(result[0].manifest_path, original.manifest_path);

        let mut survivors = manifest_file_paths(&table, &result[0]).await;
        survivors.sort();
        assert_eq!(survivors, vec![
            "data/a.parquet".to_string(),
            "data/c.parquet".to_string(),
        ]);

        // Survivors are re-emitted as EXISTING entries, never re-added.
        let manifest = result[0].load_manifest(table.file_io()).await.unwrap();
        assert!(
            manifest
                .entries()
                .iter()
                .all(|e| e.status() == ManifestStatus::Existing)
        );
    }

    #[tokio::test]
    async fn test_missing_delete_path_fails_when_required() {
        let (table, _tmp_dir) = make_fs_table();
        // The manifest holds a+b; the operation wants to remove b AND a file that is
        // nowhere in the table (already rewritten away by a concurrent commit).
        let manifest = write_data_manifest(&table, &["data/a.parquet", "data/b.parquet"]).await;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/b.parquet"));
        manager.delete_file(data_file("data/vanished.parquet"));

        let err = manager
            .filter_manifests(&mut producer, &table, vec![manifest], true)
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::PreconditionFailed);
        assert!(err.to_string().contains("data/vanished.parquet"));
    }

    #[tokio::test]
    async fn test_missing_delete_path_tolerated_when_not_required() {
        let (table, _tmp_dir) = make_fs_table();
        let manifest = write_data_manifest(&table, &["data/a.parquet", "data/b.parquet"]).await;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/b.parquet"));
        manager.delete_file(data_file("data/vanished.parquet"));

        // Same situation, but the operation does not require all removals to exist.
        let result = manager
            .filter_manifests(&mut producer, &table, vec![manifest], false)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let survivors = manifest_file_paths(&table, &result[0]).await;
        assert_eq!(survivors, vec!["data/a.parquet".to_string()]);
    }

    #[tokio::test]
    async fn test_filter_errors_on_manifest_with_unknown_partition_spec() {
        let (table, _tmp_dir) = make_fs_table();
        let mut manifest = write_data_manifest(&table, &["data/a.parquet"]).await;
        // A manifest may legitimately carry a NON-default spec (spec evolution) —
        // the filter now re-emits survivors under the source spec. But a spec id
        // absent from table metadata is corrupt input and must error.
        manifest.partition_spec_id = 99;
        let mut producer = make_producer(&table);

        let mut manager = ManifestFilterManager::default();
        manager.delete_file(data_file("data/a.parquet"));

        let err = manager
            .filter_manifests(&mut producer, &table, vec![manifest], true)
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.to_string().contains("Partition spec 99 not found"));
    }
}
