#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_cluster_server.py

맥북 로컬에서 동작하는 "이미지 클러스터링 전용" 서버.
- new_deep_clusterer.DeepClusterer 를 내부에서 사용
- HTTP API 로 이미지 경로 리스트를 전달받아 클러스터링 수행
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

# new_deep_clusterer.py 가 같은 디렉토리에 있다고 가정
from domain.pipeline import PhotoClusteringPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("image_cluster_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------------------
# Pydantic 모델 정의
# ------------------------------------------------------------------------------

class ClusterRequest(BaseModel):
    photo_paths: list[str] = Field(
        ..., description="클러스터링할 이미지의 절대 경로(or 신뢰 가능한 로컬 경로) 리스트"
    )
    similarity_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="전역 임베딩 코사인 유사도 임계값 (0~1)"
    )
    use_cache: bool = Field(
        True, description="특징 벡터 캐시 사용 여부"
    )
    remove_people: bool = Field(
        True, description="사람 영역을 마스킹할지 여부 (DETR 사용)"
    )

    @validator("photo_paths")
    def validate_paths(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("photo_paths 는 비어 있을 수 없습니다.")
        return v


class ClusterGroupResponse(BaseModel):
    id: int
    photos: List[str]
    count: int
    avg_similarity: float
    quality_score: float


class ClusterResponse(BaseModel):
    clusters: List[ClusterGroupResponse]
    total_photos: int
    total_clusters: int
    similarity_threshold: float


# ------------------------------------------------------------------------------
# FastAPI 앱 및 글로벌 DeepClusterer 인스턴스
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Local Image Cluster Server",
    description="맥북 로컬에서 동작하는 이미지 클러스터링 전용 서버 (new_deep_clusterer 기반)",
    version="1.0.0",
)

# DeepClusterer 는 모델 로딩이 무거우므로, 앱 시작 시 1회 초기화해서 재사용
# input_path 는 캐시/결과용 베이스 디렉터리만 의미하므로, 실제 이미지 위치와는 독립적.
BASE_DIR = Path(os.environ.get("IMAGE_CLUSTER_BASE_DIR", ".")).resolve()
CACHE_BASE = BASE_DIR / "cluster_cache"
if not CACHE_BASE.is_dir():
    os.makedirs(CACHE_BASE, exist_ok=True)
    logger.info(f"Created directory: {CACHE_BASE}")

# asyncio Lock 으로 한 번에 하나의 클러스터링 작업만 수행 (모델/상태 공유 보호)
clusterer_lock = asyncio.Lock()

# 앱 시작 시 초기화될 전역 인스턴스
pipeline: PhotoClusteringPipeline | None = None


@app.on_event("startup")
async def startup_event():
    global pipeline

    # 여기서는 device 선택을 new_deep_clusterer 내부에 맡김
    # (mps / cuda / cpu 중 가능한 것 자동 선택하는 구조로 만들어 두었음)
    logger.info("🔧 Initializing Pipeline for image clustering server...")
    pipeline = PhotoClusteringPipeline(CACHE_BASE=CACHE_BASE)
    logger.info("✅ PhotoClusteringPipeline initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down image cluster server...")


# ------------------------------------------------------------------------------
# 엔드포인트 정의
# ------------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    헬스 체크 엔드포인트.
    """
    return {"status": "ok"}


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_images(req: ClusterRequest) -> ClusterResponse:
    """
    이미지 경로 리스트를 입력받아 클러스터링을 수행하는 엔드포인트.

    - photo_paths: 로컬 파일 시스템 경로들 (예: /Users/you/photos/xxx.jpg)
    - 응답: 각 클러스터의 id, 포함된 사진 경로, 개수, 평균 유사도, quality_score
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="DeepClusterer 가 초기화되지 않았습니다.")

    # 존재하지 않는 파일 체크 (기본적인 검증)
    missing_files = [p for p in req.photo_paths if not Path(p).is_file()]
    logger.info(f"Get Cluster Req {len(req.photo_paths), len(missing_files)}")
    if missing_files:
        raise HTTPException(
            status_code=400,
            detail=f"다음 파일들이 존재하지 않습니다: {missing_files[:5]} "
                   f"{'(외 추가 있음 ...)' if len(missing_files) > 5 else ''}",
        )

    # 요청에서 넘어온 threshold / cache / remove_people 설정을 반영
    # (Lock 안에서 변경 -> 그 클러스터링 작업에만 유효)
    async with clusterer_lock:
        # 원래 설정 백업
        # orig_threshold = clusterer.similarity_threshold
        # orig_use_cache = clusterer.use_cache
        # orig_remove_people = clusterer.remove_people

        # clusterer.similarity_threshold = req.similarity_threshold
        # clusterer.use_cache = req.use_cache
        # clusterer.remove_people = req.remove_people

        try:
            # 실제 클러스터링 수행
            # cluster() 는 List[List[str]] (클러스터별 경로 리스트)를 반환하지만,
            # 더 자세한 정보는 clusterer.groups 에 들어 있음.
            logger.info(
                f"🚀 Clustering {len(req.photo_paths)} photos "
                f"(threshold={req.similarity_threshold}, "
                f"use_cache={req.use_cache}, remove_people={req.remove_people})"
            )

            # 동기 함수지만, 일단 그냥 호출 (CPU/GPU를 오래 점유하는 동안 이 요청은 블록됨)

            groups = await pipeline.run(req.photo_paths)

            # groups 구조에서 자세한 정보 추출
            clusters: List[ClusterGroupResponse] = []
            total_photos = 0

            for idx, g in enumerate(groups):
                # g 구조:
                # { "id", "photos", "count", "avg_similarity", "quality_score" }
                # total_photos += g["count"]
                # clusters.append(
                #     ClusterGroupResponse(
                #         id=int(g["id"]),
                #         photos=photo_paths,
                #         count=int(g['count']),
                #         avg_similarity=float(g["avg_similarity"]),
                #         quality_score=float(g["quality_score"]),
                #     )
                # )

                photo_paths = [p.path for p in g]
                total_photos += len(g)
                clusters.append(
                    ClusterGroupResponse(
                        id=idx,
                        photos=photo_paths,
                        count=int(len(g)),
                        avg_similarity=1.0,
                        quality_score=1.0,
                    )
                )
            # quality_score 기준으로 이미 정렬되어 있지만, 한 번 더 확실하게 정렬
            clusters.sort(key=lambda c: c.quality_score, reverse=True)

            resp = ClusterResponse(
                clusters=clusters,
                total_photos=total_photos,
                total_clusters=len(clusters),
                similarity_threshold=req.similarity_threshold,
            )
            logger.info(
                f"✅ Clustering done: {resp.total_clusters} clusters, "
                f"{resp.total_photos} photos."
            )
            return resp

        finally:
            # 설정 복원
            pass
            # clusterer.similarity_threshold = orig_threshold
            # clusterer.use_cache = orig_use_cache
            # clusterer.remove_people = orig_remove_people


# ------------------------------------------------------------------------------
# 개발 편의를 위한 로컬 실행 진입점
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # 예: http://127.0.0.1:8001/docs 에서 Swagger UI 확인 가능
    uvicorn.run(
        "image_cluster_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )