from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.api import api_router
from core.config import configs
from core.logger import setup_logging
from app.cluster.model_loader import initialize_all_models

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🔧 Initializing Core Engine...")
    app.state.clusterer_lock = asyncio.Lock()
    await initialize_all_models()
    logger.info("✅ Core Engine initialized.")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Core Engine...")

app = FastAPI(
    title=configs.PROJECT_NAME,
    description="Snap to Report Core Engine",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS (Allow all for development env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Snap Report Core Engine Running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
