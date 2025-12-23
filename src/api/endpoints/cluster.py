import uuid
import logging
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from core.dependencies import get_lock
from app.cluster.schema import ClusterRequest, ClusterTaskResponse
from app.cluster.services.callback_sender import CallbackSender
# from app.cluster.services.pipeline import PhotoClusteringPipeline
from app.cluster.services.clustering import ClusteringService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_clustering_service() -> ClusteringService:
    return ClusteringService(CallbackSender())


@router.post("", response_model=ClusterTaskResponse, status_code=202)
async def submit_cluster_task(
    req: ClusterRequest,
    background_tasks: BackgroundTasks,
    service: ClusteringService = Depends(get_clustering_service),
    lock: asyncio.Lock = Depends(get_lock),
):
    """
    Submit an asynchronous image clustering task.
    """
    task_id = str(uuid.uuid4())
    
    # 2. Log Acceptance
    logger.info(f"📥 [Task {task_id}] Accepted. ReqID: {req.request_id}, Webhook: {req.webhook_url}")

    # 3. Delegate to Background Task via Service
    background_tasks.add_task(
        service.process_task,
        task_id=task_id,
        req=req,
        lock=lock
    )

    # 4. Immediate Response
    return ClusterTaskResponse(
        task_id=task_id,
        request_id=req.request_id,
        status="processing",
        message="Clustering task started in background."
    )