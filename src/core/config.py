
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Snap To Report Worker"
    LOG_LEVEL: str = "DEBUG"
    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    
    # Storage
    STORAGE_TYPE: str = "gcs"
    GCS_BUCKET_NAME: str = "bucket"
    
    # PDF
    PDF_BASE_TEMPLATE_PATH: str = "templates/base_template.pdf"
    PDF_COVER_TEMPLATE_PATH: str = "templates/cover_template.pdf"

    class Config:
        env_file = ".env"

configs = Settings()
