import uuid
import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends

from core.dependencies import get_lock
from common.callback_sender import CallbackSender
from app.pdf.schema import PDFTaskResponse, PDFGenerateRequest
from app.pdf.service import PDFService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pdf_service() -> PDFService:
    return PDFService(CallbackSender())


@router.post("", response_model=PDFTaskResponse, status_code=202)
async def generate_pdf(
    req: PDFGenerateRequest,
    background_tasks: BackgroundTasks,
    service: PDFService = Depends(get_pdf_service),
    lock: asyncio.Lock = Depends(get_lock),
):
    task_id = str(uuid.uuid4())
    logger.info(f"📥 [Task {task_id}] Accepted. ReqID: {req.request_id}")

    background_tasks.add_task(
        service.process_task,
        task_id=task_id,
        req=req,
        lock=lock
    )

    return PDFTaskResponse(
        task_id=task_id,
        request_id=req.request_id,
        status="processing",
        message="PDF generation started in background.",
    )