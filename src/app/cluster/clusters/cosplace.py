import asyncio
import gc
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.storage.factory import get_storage_client
from app.utils.performance import PerformanceMonitor
from app.cluster.models import PhotoMeta
from app.cluster.clusters.base import Clusterer
from app.cluster.extractors.cosplace import CosPlaceExtractor

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as HDBSCAN
    except ImportError:
        HDBSCAN = None

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

# Parameters
PARAMS = {
    "eps": 7.488875486571053,
    "max_gps_tol": 44.35800514729294,
    "min_cluster_size": 2,
    "min_samples": 1,
    "max_cluster_size": 8,
    "strict_thresh": 0.15011476241322744,
    "loose_thresh": 0.5018121751611363,
    "w_merge": 0.10088178479325592,
    "w_split": 3.669884372778316,
}

class CosPlaceCluster(Clusterer):
    def __init__(self, extractor: Optional[CosPlaceExtractor] = None):
        self.params = PARAMS

        self.visual_split_thresh = self.params.get("loose_thresh", 0.64)
        self.split_min_size = 8
        self.min_cluster_size = self.params.get("min_cluster_size", 2)
        self.min_samples = self.params.get("min_samples", 1)
        self.max_cluster_size = self.params.get("max_cluster_size", 4)

        self.extractor = extractor if extractor is not None else CosPlaceExtractor(onnx_path="img_models/cosplace_resnet50_int8.onnx")
        self.storage = get_storage_client()
        self.performance_monitor = PerformanceMonitor()

        logger.info(f"CosPlaceCluster Ready. Mode: {'ONNX' if self.extractor.use_onnx else 'Torch/None'}")

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        logger.info(f"CosPlaceCluster Mode: {'ONNX' if self.extractor.use_onnx else 'Torch/None'}")
        self.performance_monitor.start()

        logger.info(f"Start feature extraction for {len(photos)} photos.")
        features_matrix = await self._extract_features_optimized(photos)
        labels = self._run_clustering_logic(photos, features_matrix)

        clusters = {}
        noise = []
        for i, label in enumerate(labels):
            if label == -1:
                noise.append(photos[i])
            else:
                clusters.setdefault(label, []).append(photos[i])

        result = list(clusters.values())
        if noise:
            result.append(noise)

        self.performance_monitor.stop()
        self.performance_monitor.report("CosPlaceCluster", count=len(photos))

        return result

    async def _extract_features_optimized(self, photos: List[PhotoMeta]) -> List[Optional[np.ndarray]]:
        """
        Batch Size만큼 다운로드 -> 메모리 로드 -> Batch 추론 -> 메모리 해제
        반환값: 각 사진에 해당하는 Feature Vector (List 순서 보장)
        """
        n_photos = len(photos)
        final_features = [None] * n_photos

        BATCH_SIZE = 16
        semaphore = asyncio.Semaphore(10) 

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i in range(0, n_photos, BATCH_SIZE):
                batch_indices = range(i, min(i + BATCH_SIZE, n_photos))
                batch_photos = [photos[idx] for idx in batch_indices]

                download_tasks = []
                local_files_map = {}  # idx -> file_path

                for idx, p in zip(batch_indices, batch_photos):
                    target_path = p.thumbnail_path or p.path
                    if not target_path:
                        if os.path.exists(p.path):
                            local_files_map[idx] = Path(p.path)
                        continue

                    ext = os.path.splitext(target_path)[1].split("?")[0] or ".jpg"
                    dest_path = temp_path / f"{idx}{ext}"
                    local_files_map[idx] = dest_path

                    download_tasks.append(self._download_safe(target_path, dest_path, semaphore))

                if download_tasks:
                    await asyncio.gather(*download_tasks)

                # B. 메모리에 이미지 로드 (Batch 준비)
                valid_indices = []
                pil_images = []

                for idx in batch_indices:
                    if idx in local_files_map and local_files_map[idx].exists():
                        try:
                            with Image.open(local_files_map[idx]) as img:
                                # [Memory Optimization] Resize immediately to (640, 480)
                                # This prevents OOM when holding multiple high-res photos in 'pil_images'
                                img_resized = img.convert("RGB").resize((640, 480))
                                pil_images.append(img_resized)
                                valid_indices.append(idx)
                        except Exception as e:
                            logger.warning(f"Failed to open image {idx}: {e}")

                if pil_images:
                    batch_features = self.extractor.extract_batch(pil_images)

                    for sub_idx, feature in enumerate(batch_features):
                        original_idx = valid_indices[sub_idx]
                        final_features[original_idx] = feature

                del pil_images
                del batch_photos
                for fpath in local_files_map.values():
                    if fpath.exists():
                        fpath.unlink(missing_ok=True)
                
                gc.collect()

        return final_features

    async def _download_safe(self, url: str, dest: Path, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                await self.storage.download_file(url, dest)
            except Exception as e:
                logger.warning(f"Download error {url}: {e}")

    def _run_clustering_logic(
        self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]],
    ) -> np.ndarray:

        n_samples = len(photos)
        # None Feature가 있는지 확인하고 필터링하거나 0벡터 처리 (여기서는 인덱스 유지를 위해 None 체크)
        valid_mask = [f is not None for f in features]
        if not any(valid_mask):
            return np.full(n_samples, -1)

        if HDBSCAN is None:
            return np.full(n_samples, -1)

        labels = np.zeros(n_samples, dtype=int)
        max_label = labels.max()
        next_label_id = max_label + 1

        unique_labels = set(labels) - {-1}

        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) <= self.split_min_size:
                continue

            sub_features = []
            valid_sub_indices = []

            for idx in indices:
                if features[idx] is not None:
                    sub_features.append(features[idx])
                    valid_sub_indices.append(idx)

            if len(sub_features) < 2:
                continue

            sub_features_arr = np.array(sub_features)  # (N, 512)

            try:
                sub_clusterer = HDBSCAN(
                    min_cluster_size=2,
                    min_samples=2,
                    metric="euclidean",
                    cluster_selection_epsilon=self.visual_split_thresh,
                    allow_single_cluster=False,
                )
                sub_labels = sub_clusterer.fit_predict(sub_features_arr)

                found_subs = set(sub_labels) - {-1}
                for sub_id in found_subs:
                    # 원본 indices 중, 현재 sub_id에 해당하는 것들만 필터링
                    # 주의: sub_labels 길이는 valid_sub_indices 길이와 같음
                    mask = sub_labels == sub_id
                    target_real_indices = np.array(valid_sub_indices)[mask]

                    labels[target_real_indices] = next_label_id
                    next_label_id += 1

            except Exception as e:
                logger.warning(f"Visual split failed for cluster {cluster_id}: {e}")

        # Force Split (K-Means): 최대 크기 초과시 강제 분할
        unique_labels = set(labels) - {-1}
        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > self.max_cluster_size:

                sub_features = []
                valid_sub_indices = []
                for idx in indices:
                    if features[idx] is not None:
                        sub_features.append(features[idx])
                        valid_sub_indices.append(idx)

                if len(sub_features) < self.max_cluster_size:
                    continue

                n_splits = math.ceil(len(valid_sub_indices) / self.max_cluster_size)

                try:
                    kmeans = KMeans(n_clusters=n_splits, n_init=10, random_state=42)
                    sub_labels = kmeans.fit_predict(np.stack(sub_features))

                    for k in range(n_splits):
                        mask = sub_labels == k
                        target_real_indices = np.array(valid_sub_indices)[mask]
                        labels[target_real_indices] = next_label_id
                        next_label_id += 1

                except Exception as e:
                    logger.warning(f"KMeans split failed for cluster {cluster_id}: {e}")
        return labels