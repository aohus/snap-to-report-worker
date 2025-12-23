
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Snap Report Core"
    LOG_LEVEL: str = "DEBUG"
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    
    # Storage
    STORAGE_TYPE: str = "gcs" # local, gcs, s3
    MEDIA_ROOT: str = "media"
    MEDIA_URL: str = "/media"
    GCS_BUCKET_NAME: str = "snap-2-report-assets-1764756531"

    
    # PDF
    PDF_BASE_TEMPLATE_PATH: str = "templates/base_template.pdf"
    
    # Cluster Service
    CLUSTER_SERVICE_URL: str = "http://localhost:8000"
    CALLBACK_BASE_URL: str = "http://host.docker.internal:8000/api"

    class Config:
        env_file = ".env"

configs = Settings()
