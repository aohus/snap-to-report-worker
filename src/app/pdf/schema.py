from pydantic import BaseModel


class PDFPhoto(BaseModel):
    id: str
    path: str
    timestamp: str
    labels: dict

class PDFCluster(BaseModel):
    id: str
    title: str
    photos: list[PDFPhoto]

class PDFGenerateRequest(BaseModel):
    request_id: str
    bucket_path: str
    cover_title: str = ""
    cover_company_name: str = ""
    label_config: dict = {}
    webhook_url: str | None = None
    clusters: list[PDFCluster]

class PDFTaskResponse(BaseModel):
    task_id: str
    request_id: str
    status: str
    message: str
