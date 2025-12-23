from api.endpoints import cluster, pdf
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(cluster.router, prefix="/cluster", tags=["Clustering Similar Photos"])
api_router.include_router(pdf.router, prefix="/pdf", tags=["Generate Pdf"])