import asyncio
import logging
import traceback
from pathlib import Path
from typing import List

from app.cluster.schema import ClusterRequest, ClusterResponse
from app.cluster.services.callback_sender import CallbackSender
from app.cluster.services.pipeline import PhotoClusteringPipeline
from app.cluster.services.formatters import format_cluster_response
from core.storage.factory import get_storage_client
from core.storage.local import LocalStorageService
from app.config import JobConfig, ClusteringConfig
from app.common.models import Photo

logger = logging.getLogger(__name__)


class ClusteringService:
    def __init__(self, callback_sender: CallbackSender):
        # self.pipeline dependency removed
        self.callback_sender = callback_sender

    async def process_task(
        self,
        task_id: str,
        req: ClusterRequest,
        lock: asyncio.Lock
    ):
        """백그라운드 작업 메인 로직"""
        logger.info(f"🏁 [Task {task_id}] Background processing started. Bucket: {req.bucket_path}")

        try:
            # 1. List files from storage
            storage = get_storage_client()
            files = await storage.list_files(req.bucket_path)
            
            if not files:
                raise ValueError(f"No files found in bucket path: {req.bucket_path}")

            # 2. Prepare Photos for Pipeline
            photos: List[Photo] = []
            if isinstance(storage, LocalStorageService):
                for f in files:
                    # f is relative path from media_root
                    full_path = storage.media_root / f
                    photos.append(Photo(storage_path=str(f), original_filename=Path(f).name))
            else:
                for f in files:
                    # f is object key
                    photos.append(Photo(storage_path=f, original_filename=Path(f).name))

            logger.info(f"✅ [Task {task_id}] Found {len(photos)} photos.")

            # 3. Configure Pipeline
            # JobConfig creation
            # Note: req.cluster_job_id might be None if client doesn't send it, fallback to request_id
            job_id = req.cluster_job_id or req.request_id
            
            clustering_config = ClusteringConfig(
                # Override defaults if needed based on req (e.g. min_samples)
                # gps=GPSConfig(min_samples=req.min_samples) # if we parse it
            )
            # Assuming min_samples is handled by configs or passed down if implemented
            
            job_config = JobConfig(
                job_id=job_id,
                clustering=clustering_config
            )

            # 4. Pipeline Execution (Concurrency Control)
            async with lock:
                pipeline = PhotoClusteringPipeline(job_config, storage, photos)
                final_clusters = await pipeline.run()

            # 5. Response Formatting
            response_clusters = format_cluster_response(final_clusters)
            
            # 통계 계산
            total_photos = sum(c.count for c in response_clusters)
            
            result_payload = ClusterResponse(
                clusters=response_clusters,
                total_photos=total_photos,
                total_clusters=len(response_clusters),
                similarity_threshold=req.similarity_threshold
            ).dict()

            full_payload = {
                "task_id": task_id,
                "request_id": req.request_id,
                "status": "completed",
                "result": result_payload
            }

            logger.info(f"✅ [Task {task_id}] Completed. Total clusters: {len(response_clusters)}")

            # 6. Callback
            if req.webhook_url:
                await self.callback_sender.send_result(str(req.webhook_url), full_payload, task_id)
            else:
                logger.warning(f"⚠️ [Task {task_id}] No webhook_url. Result not persisted.")

        except Exception as e:
            logger.error(f"💥 [Task {task_id}] Failed: {e}")
            logger.debug(traceback.format_exc())
            
            if req.webhook_url:
                await self.callback_sender.send_error(
                    str(req.webhook_url), str(e), task_id, req.request_id
                )