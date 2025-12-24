import asyncio
import logging
import time
from typing import Dict, List

from app.config import JobConfig
from app.cluster.clusters.base import Clusterer
from app.cluster.clusters.cosplace import CosPlaceCluster 
from app.cluster.model_loader import get_cos_place_extractor #, get_cos_place_extractor_legacy
from app.cluster.services.metadata_extractor import MetadataExtractor

# if TYPE_CHECKING:
#     from core.storage.base import StorageService
from core.storage.base import StorageService
from core.storage.local import LocalStorageService
from app.models.photometa import Photo, PhotoMeta

logger = logging.getLogger(__name__)


class ClusterRunner:
    def __init__(self, clusterers: List[Clusterer]):
        self.clusterers = clusterers

    async def process(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Processes photos by applying a series of clustering clusterers in sequence.
        """
        # Start with a single cluster containing all photos
        clusters = [photos]

        for clusterer in self.clusterers:
            logger.info(f"Applying clusterer: {clusterer.__class__.__name__}")

            new_clusters = []
            # Apply the clusterer to each existing cluster
            for cluster in clusters:
                if not cluster:
                    continue
                if clusterer.condition(cluster):
                    sub_clusters = await clusterer.cluster(cluster)
                    new_clusters.extend(sub_clusters)
                else:
                    new_clusters.append(cluster)

            clusters = new_clusters
            logger.info(f"Resulted in {len(clusters)} clusters.")
        return clusters


class PhotoClusteringPipeline:
    def __init__(self, config: JobConfig, storage: StorageService, photos: list[Photo]):
        self.config = config
        self.storage = storage
        self.photos = photos
        logger.debug(f"Initializing pipeline for job_id: {self.config.job_id}")

        self.metadata_extractor = MetadataExtractor()

        clusterers = self._create_clusterers(self.config)
        self.clusterer = ClusterRunner(clusterers)
        logger.debug(f"Pipeline initialized with {len(clusterers)} clusterers.")

    def _create_clusterers(self, config: "JobConfig") -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        logger.debug("Creating clusterers...")
        
        extractor = get_cos_place_extractor()
        return [CosPlaceCluster(extractor=extractor)]

    async def run(self):
        logger.info(f"Pipeline run started for job {self.config.job_id}.")

        logger.info("Resolving photo paths and extracting metadata...")

        extracted_metas = [None] * len(self.photos)
        tasks = []
        indices_to_extract = []

        for i, p in enumerate(self.photos):
            if p.meta_lat is not None and p.meta_lon is not None:
                extracted_metas[i] = PhotoMeta(
                    path=p.storage_path,
                    thumbnail_path=p.thumbnail_path,
                    original_name=p.original_filename,
                    device=p.device,
                    lat=p.meta_lat,
                    lon=p.meta_lon,
                    alt=None,  # Altitude not stored in DB yet
                    timestamp=p.meta_timestamp.timestamp() if p.meta_timestamp else None,
                    orientation=None,
                    digital_zoom=None,
                    scene_capture_type=None,
                    white_balance=None,
                    exposure_mode=None,
                    flash=None,
                    gps_img_direction=None,
                )
            else:
                indices_to_extract.append(i)
                if isinstance(self.storage, LocalStorageService):
                    full_path = str(self.storage.media_root / p.storage_path)
                else:
                    full_path = self.storage.get_url(p.storage_path)
                tasks.append(self.metadata_extractor.extract(full_path))

        if tasks:
            logger.info(f"Extracting metadata for {len(tasks)} photos (others used cache)...")
            results = await asyncio.gather(*tasks)
            for idx, meta in zip(indices_to_extract, results):
                extracted_metas[idx] = meta
        else:
            logger.info("Using cached metadata for all photos.")

        photos = [m for m in extracted_metas if m is not None]
        logger.info(f"Metadata ready for {len(photos)} photos.")

        logger.info("Starting clustering pipeline...")
        final_scenes = await self.clusterer.process(photos)
        logger.info(f"Clustering pipeline finished. Final scene count: {len(final_scenes)}")

        if not final_scenes:
            logger.warning("No scenes were generated. Skipping output generation.")
            return
        return final_scenes
