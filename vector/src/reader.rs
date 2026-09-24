//! Read-only client for the vector database.
//!
//! `VectorDbReader` provides a lightweight, read-only interface to a vector
//! database. Unlike `VectorDb`, it does not participate in fencing and does
//! not hold a write lease, so multiple readers can coexist with a single
//! writer.

use crate::db::VectorDbRead;
use crate::error::{Error, Result};
use crate::model::{Query, ReaderConfig, SearchOptions, SearchResult};
use crate::query_engine::{QueryEngine, QueryEngineOptions};
use crate::serde::vector_bitmap::VectorBitmap;
use crate::storage::merge_operator::VectorDbMergeOperator;
use crate::storage::VectorDbStorageReadExt;
use crate::write::indexer::tree::centroids::{
    LeveledCentroidIndex, StoredCentroidReader, TreeDepth,
};
use crate::Vector;
use async_trait::async_trait;
use common::storage::factory::{create_storage_read, StorageReaderRuntime};
use common::storage::StorageRead;
use common::StorageConfig;
use common::StorageSemantics;
use std::sync::Arc;

/// Read-only client for querying a vector database.
///
/// `VectorDbReader` loads a vector db from storage for read-only access.
/// NOTE: currently the reader only works for dbs that are static as it does not
///       have a mechanism for refreshing the centroid graph. This is deferred to
///       a later improvement. In the interim we'll first add support for creating
///       a reader from a checkpoint.
///
pub struct VectorDbReader {
    options: QueryEngineOptions,
    centroid_index: Arc<LeveledCentroidIndex<'static>>,
    storage: Arc<dyn StorageRead>,
    /// FTS deletions bitmap loaded at open time. The reader only supports
    /// static dbs, so this is loaded once and shared with each QueryEngine.
    deletions: Arc<VectorBitmap>,
}

/// Copy the settings-file knobs the cell writes (`manifest_poll_interval`,
/// native object-store cache) onto the reader options. In-memory storage and
/// a missing `settings_path` keep [`DbReaderOptions::default`].
fn db_reader_options(storage: &StorageConfig) -> Result<slatedb::config::DbReaderOptions> {
    let mut options = slatedb::config::DbReaderOptions::default();
    let StorageConfig::SlateDb(slate) = storage else {
        return Ok(options);
    };
    let Some(path) = &slate.settings_path else {
        return Ok(options);
    };
    let settings = slatedb::config::Settings::from_file(path)
        .map_err(|err| Error::Storage(format!("reader settings {path}: {err}")))?;
    options.manifest_poll_interval = settings.manifest_poll_interval;
    options.object_store_cache_options = settings.object_store_cache_options;
    Ok(options)
}

impl VectorDbReader {
    /// Open a read-only client against an existing vector database.
    ///
    /// Loads the centroid graph from storage. The database must have been
    /// previously initialized by a `VectorDb` writer.
    pub async fn open(config: ReaderConfig) -> Result<Self> {
        Self::open_with_runtime(config, StorageReaderRuntime::new()).await
    }

    /// Open a read-only client with custom runtime options (e.g. block cache).
    pub async fn open_with_runtime(
        config: ReaderConfig,
        runtime: StorageReaderRuntime,
    ) -> Result<Self> {
        let merge_op = VectorDbMergeOperator::new(config.dimensions as usize);
        let storage = create_storage_read(
            &config.storage,
            runtime,
            StorageSemantics::new()
                .with_merge_operator(Arc::new(merge_op))
                .with_segment_extractor(
                    crate::storage::segment_extractor::VectorSegmentExtractor::shared(),
                ),
            db_reader_options(&config.storage)?,
        )
        .await?;

        let dimensions = config.dimensions as usize;
        let centroids_meta = storage.get_centroids_meta().await?;
        if centroids_meta.is_none() {
            return Err(Error::Storage(
                "No centroid tree found in storage. Database must be initialized by VectorDb first."
                    .to_string(),
            ));
        }
        let reader = Arc::new(StoredCentroidReader::new(dimensions, storage.clone(), 0));
        let centroid_index = Arc::new(LeveledCentroidIndex::new(
            TreeDepth::of(centroids_meta.expect("checked above").depth),
            config.distance_metric,
            reader,
        ));

        let options = QueryEngineOptions {
            dimensions: config.dimensions,
            distance_metric: config.distance_metric,
            query_pruning_factor: config.query_pruning_factor,
        };

        let deletions = Arc::new(storage.get_deletions().await?);

        Ok(Self::new(options, centroid_index, storage, deletions))
    }

    pub(crate) fn new(
        options: QueryEngineOptions,
        centroid_index: Arc<LeveledCentroidIndex<'static>>,
        storage: Arc<dyn StorageRead>,
        deletions: Arc<VectorBitmap>,
    ) -> Self {
        Self {
            options,
            centroid_index,
            storage,
            deletions,
        }
    }

    /// Close the reader, releasing any storage-side resources.
    ///
    /// For SlateDB-backed storage this releases the reader handle and
    /// flushes the SlateDB reader runtime; for in-memory storage this is
    /// a no-op. Prefer calling `close` over relying on `Drop` so the
    /// shutdown happens deterministically while the async runtime is
    /// still alive.
    pub async fn close(&self) -> Result<()> {
        self.storage.close().await?;
        Ok(())
    }

    fn query_engine(&self) -> QueryEngine {
        QueryEngine::new(
            self.options.clone(),
            self.centroid_index.clone(),
            self.storage.clone(),
            self.deletions.clone(),
        )
    }
}

#[async_trait]
impl VectorDbRead for VectorDbReader {
    async fn search_with_options(
        &self,
        query: &Query,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        self.query_engine()
            .search_with_options(query, options)
            .await
    }

    async fn get(&self, id: &str) -> Result<Option<Vector>> {
        self.query_engine().get(id).await
    }
}

#[cfg(test)]
mod tests {
    use crate::db::VectorDbRead;
    use crate::model::{Config, Query, ReaderConfig, Vector};
    use crate::reader::VectorDbReader;
    use crate::serde::collection_meta::DistanceMetric;
    use crate::VectorDb;
    use common::storage::config::{
        LocalObjectStoreConfig, ObjectStoreConfig, SlateDbStorageConfig,
    };
    use common::StorageConfig;
    use std::time::Duration;
    use tempfile::TempDir;

    fn local_storage_config(dir: &TempDir) -> StorageConfig {
        StorageConfig::SlateDb(SlateDbStorageConfig {
            path: "vector-data".to_string(),
            object_store: ObjectStoreConfig::Local(LocalObjectStoreConfig {
                path: dir.path().to_string_lossy().to_string(),
            }),
            settings_path: None,
            block_cache: None,
            meta_cache: None,
        })
    }

    #[tokio::test]
    async fn should_search_vectors_via_reader() {
        // given - write vectors via VectorDb
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = local_storage_config(&temp_dir);

        let config = Config {
            storage: storage.clone(),
            dimensions: 3,
            distance_metric: DistanceMetric::L2,
            flush_interval: Duration::from_secs(60),
            split_threshold_vectors: 10_000,
            ..Default::default()
        };
        let db = VectorDb::open(config).await.unwrap();

        let vectors = vec![
            Vector::new("vec-1", vec![1.0, 0.0, 0.0]),
            Vector::new("vec-2", vec![0.0, 1.0, 0.0]),
            Vector::new("vec-3", vec![0.0, 0.0, 1.0]),
        ];
        db.write(vectors).await.unwrap();
        db.flush().await.unwrap();

        // when - open a reader and search
        let reader_config = ReaderConfig {
            storage,
            dimensions: 3,
            distance_metric: DistanceMetric::L2,
            query_pruning_factor: None,
            metadata_fields: vec![],
        };
        let reader = VectorDbReader::open(reader_config).await.unwrap();
        let results = reader
            .search(&Query::new(vec![1.0, 0.0, 0.0]).with_limit(2))
            .await
            .unwrap();

        // then - closest vector should be vec-1
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector.id, "vec-1");
    }

    #[test]
    fn settings_file_sets_poll_and_object_store_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reader.json");
        std::fs::write(
            &path,
            r#"{
              "manifest_poll_interval": "1500ms",
              "object_store_cache_options": {
                "root_folder": "/data/native-osc",
                "max_cache_size_bytes": 1048576,
                "cache_puts": false,
                "preload_disk_cache_on_startup": "AllSst",
                "scan_interval": "15s"
              }
            }"#,
        )
        .unwrap();
        let storage = StorageConfig::SlateDb(SlateDbStorageConfig {
            path: "data".into(),
            object_store: ObjectStoreConfig::Local(LocalObjectStoreConfig {
                path: dir.path().to_str().unwrap().into(),
            }),
            settings_path: Some(path.display().to_string()),
            block_cache: None,
            meta_cache: None,
        });
        let options = super::db_reader_options(&storage).unwrap();
        assert_eq!(options.manifest_poll_interval, Duration::from_millis(1500));
        assert_ne!(
            options.manifest_poll_interval,
            slatedb::config::DbReaderOptions::default().manifest_poll_interval
        );
        let cache = options.object_store_cache_options;
        assert_eq!(
            cache
                .root_folder
                .as_deref()
                .map(|p| p.display().to_string())
                .as_deref(),
            Some("/data/native-osc")
        );
        assert_eq!(cache.max_cache_size_bytes, Some(1_048_576));
        assert!(!cache.cache_puts);
        assert_eq!(cache.scan_interval, Some(Duration::from_secs(15)));
    }
}
