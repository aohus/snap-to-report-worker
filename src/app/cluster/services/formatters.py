from typing import List
from app.cluster.schema import ClusterGroupResponse, ClusterPhoto
# app.models.photometa 같은 내부 모델 import 가정

def _to_cluster_photo(p) -> ClusterPhoto:
    """내부 Photo 객체를 응답용 Pydantic 모델로 변환"""
    return ClusterPhoto(
        path=p.path,
        timestamp=p.timestamp,
        lat=p.lat,
        lon=p.lon
    )

def format_cluster_response(final_clusters: List[List], noise_id: int = -1) -> List[ClusterGroupResponse]:
    """
    클러스터링 결과를 API 응답 포맷으로 변환 및 노이즈 분리
    """
    formatted_clusters = []
    noise_photos = []

    # 1. 유효 클러스터와 노이즈(1장짜리) 분리
    valid_clusters = []
    for cluster in final_clusters:
        if len(cluster) == 1:
            noise_photos.extend(cluster)
        else:
            valid_clusters.append(cluster)

    # 2. 유효 클러스터 포매팅
    for idx, cluster in enumerate(valid_clusters):
        photo_details = [_to_cluster_photo(p) for p in cluster]
        formatted_clusters.append(
            ClusterGroupResponse(
                id=idx,
                photos=[p.path for p in cluster],
                photo_details=photo_details,
                count=len(cluster),
                avg_similarity=1.0,  # 추후 로직 구현 시 대체
                quality_score=1.0,
            )
        )

    # 3. 노이즈 그룹 처리
    if noise_photos:
        noise_details = [_to_cluster_photo(p) for p in noise_photos]
        formatted_clusters.append(
            ClusterGroupResponse(
                id=noise_id,
                photos=[p.path for p in noise_photos],
                photo_details=noise_details,
                count=len(noise_photos),
                avg_similarity=0.0,
                quality_score=0.0
            )
        )

    return formatted_clusters