import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.pdf.service import generate_pdf_for_session

logger = logging.getLogger(__name__)
router = APIRouter()


class PDFPhoto(BaseModel):
    id: str
    url: str
    timestamp: str
    labels: dict

class PDFCluster(BaseModel):
    id: str
    title: str
    photos: list[PDFPhoto]

class PDFGenerateRequest(BaseModel):
    export_job_id: str
    bucket_path: str
    cover_title: str
    cover_company_name: str
    clusters: list[PDFCluster]


@router.post("", status_code=202)
async def generate_pdf(
    req: PDFGenerateRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received PDF generation request for export_job_id: {req.export_job_id}")
    background_tasks.add_task(generate_pdf_for_session, req.export_job_id)
    return {"status": "accepted", "export_job_id": req.export_job_id}
