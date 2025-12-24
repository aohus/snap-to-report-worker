from pydantic import BaseModel


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

class PDFTaskResponse(BaseModel):
    task_id: str
    request_id: str
    status: str
    message: str
