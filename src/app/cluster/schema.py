from typing import Any, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ClusterRequest(BaseModel):
    webhook_url: str = Field(description="Web hook URL")
    bucket_path: str = Field(description="bucket path")
    min_samples: int = Field(3, description="min samples")
    cluster_job_id: Optional[str] = Field(None, description="Client tracking ID")
    photo_cnt: int = Field(description="Photo count")
    request_id: str = Field(..., description="Request ID")
    use_cache: bool = Field(True, description="Cache usage")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Similarity threshold")


class ClusterPhoto(BaseModel):
    path: str
    timestamp: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ClusterGroupResponse(BaseModel):
    id: int
    photos: list[str]
    photo_details: Optional[list[ClusterPhoto]] = None
    count: Optional[int] = None
    is_noise: Optional[bool] = None
    avg_similarity: float
    quality_score: float


class ClusterResponse(BaseModel):
    clusters: list[ClusterGroupResponse]
    total_photos: int
    total_clusters: int
    similarity_threshold: float


class ClusterTaskResponse(BaseModel):
    task_id: str
    status: str = "processing"
    message: str = "Clustering task accepted."
    request_id: Optional[str] = None
