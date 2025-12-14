import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor  # I/O 바운드 작업에 적합
from typing import Dict, List, Optional

import numpy as np

# GCP Libraries
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from PIL import Image, ImageFile
from pyproj import Geod

from app.domain.clusterers.base import Clusterer
from app.models.photometa import PhotoMeta

# 사용자 정의 모듈
from experiment.optimize_features.clusters import _extract_vertex_feature

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    import hdbscan as HDBSCAN

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True



class HybridCluster(Clusterer):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        self.max_gps_tolerance_m = 50.0 
        
        # Vertex AI 임베딩 기준 임계값 (Cosine Distance)
        # 0.0 (동일) ~ 1.0 (다름) ~ 2.0 (정반대)
        # Vertex AI는 의미론적 유사도가 매우 정확하므로 임계값을 타이트하게 잡아도 됨
        self.similarity_strict_thresh = 0.15  # 거의 같은 장소 (계절/공사 유무만 다름)
        self.similarity_loose_thresh = 0.35   # 확실히 다른 장소

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        # 1. GPS 보정
        self._correct_outliers_by_speed(photos)
        self._adjust_gps_inaccuracy(photos)
        
        valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]
        
        if not valid_photos: return [photos]
        
        # 2. Vertex AI 임베딩 추출 (ThreadPool 사용)
        img_paths = [p.path for p in valid_photos]
        logger.info(f"Extracting Vertex AI embeddings for {len(valid_photos)} photos...")
        
        features = []
        if img_paths:
            # API 호출은 I/O 작업이므로 ThreadPool이 훨씬 빠르고 효율적임 (CPU 안 씀)
            # max_workers를 높여서(예: 10~20) 병렬로 API를 쏘면 10초 내 처리 가능
            with ThreadPoolExecutor(max_workers=20) as executor:
                features = list(executor.map(_extract_vertex_feature, img_paths))
        else:
            features = [None] * len(valid_photos)

        # 3. 가중치 거리 행렬 계산
        dist_matrix = self._compute_weighted_distance_matrix(valid_photos, features)
        
        # 4. HDBSCAN 적용
        try:
            clusterer = HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric='precomputed',
                cluster_selection_epsilon=3.0, # 3.0 Weighted Meter
                cluster_selection_method='leaf'
            )
            labels = clusterer.fit_predict(dist_matrix)
        except Exception as e:
            logger.error(f"HDBSCAN error: {e}")
            return [valid_photos]
        
        # 5. 결과 정리
        clusters = self._group_by_labels(valid_photos, labels)
        if no_gps_photos: clusters.append(no_gps_photos)
        return clusters

    def _compute_weighted_distance_matrix(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat, p.lon] for p in photos])
        
        for i in range(n):
            for j in range(i + 1, n):
                _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                
                weight_factor = 1.0 
                
                if features[i] is not None and features[j] is not None:
                    # Cosine Distance (이미 정규화되었으므로 Dot Product만 하면 됨)
                    # 범위: 0(동일) ~ 1(직교) ~ 2(반대)
                    similarity = np.dot(features[i], features[j])
                    struct_dist = 1.0 - similarity
                    
                    # Vertex AI 모델 신뢰도 기반 가중치
                    
                    # Case A: 의미적으로 매우 유사함 (공사 전후, 계절 변화 등은 AI가 '유사'하다고 판단함)
                    if struct_dist < self.similarity_strict_thresh: 
                        # GPS 거리 10%로 축소 -> 강력하게 병합
                        weight_factor = 0.1 
                        
                    # Case B: 의미적으로 다름 (다른 건물, 다른 도로)
                    elif struct_dist > self.similarity_loose_thresh:
                        # GPS 거리 5배 확대 -> 강력하게 분리
                        weight_factor = 5.0 + (struct_dist - 0.35) * 10.0
                    
                    # Case C: 애매함
                    else:
                        weight_factor = 1.0 + (struct_dist - 0.2) * 3.0
                
                else:
                    # 임베딩 실패 시 GPS 의존
                    if gps_dist < 10.0: weight_factor = 1.0
                    else: weight_factor = 2.0

                final_dist = gps_dist * weight_factor
                
                # GPS 물리적 한계 필터
                if gps_dist > self.max_gps_tolerance_m:
                    # AI가 99% 확신하는 경우(0.1)가 아니면 50m 밖은 다른 장소
                    if weight_factor > 0.15: 
                        final_dist = 1000.0

                dist_matrix[i][j] = dist_matrix[j][i] = final_dist
                
        return dist_matrix

    def _correct_outliers_by_speed(self, photos): 
        # (기존 코드 유지)
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        max_speed_mps = 5.0 
        for i in range(1, len(timed_photos)):
            prev = timed_photos[i-1]
            curr = timed_photos[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0: continue
            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
            if (dist / dt) > max_speed_mps:
                curr.lat = prev.lat
                curr.lon = prev.lon
                if prev.alt is not None: curr.alt = prev.alt

    def _adjust_gps_inaccuracy(self, photos): 
        # (기존 코드 유지)
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        for i in range(len(timed_photos) - 2, -1, -1):
            p1 = timed_photos[i]
            p2 = timed_photos[i+1]
            if 0 <= (p2.timestamp - p1.timestamp) <= 20:
                if p2.lat is not None and p2.lon is not None:
                    p1.lat = p2.lat
                    p1.lon = p2.lon
                    if p2.alt is not None: p1.alt = p2.alt

    def _group_by_labels(self, photos, labels):
        # (기존 코드 유지)
        clusters = {}
        noise = []
        for p, label in zip(photos, labels):
            if label == -1: noise.append(p)
            else: clusters.setdefault(label, []).append(p)
        result = list(clusters.values())
        if noise: 
            for n_photo in noise: result.append([n_photo])
        return result



# 기존 HybridCluster 로직을 가져오되, 파라미터를 __init__에서 받도록 수정
class TunableHybridCluster:
    def __init__(self, params: dict):
        self.geod = Geod(ellps="WGS84")
        
        # Optuna가 제안한 파라미터들
        self.strict_thresh = params['strict_thresh'] # 예: 0.15
        self.loose_thresh = params['loose_thresh']   # 예: 0.35
        self.eps = params['eps']                     # 예: 3.0
        self.max_gps_tol = params['max_gps_tol']     # 예: 50.0
        
        # 가중치 강도 조절 계수 (튜닝 대상)
        self.w_merge = params.get('w_merge', 0.1)  # 병합 시 거리 축소 비율
        self.w_split = params.get('w_split', 5.0)  # 분리 시 거리 확대 비율

    def run_clustering(self, photos, features):
        """
        이미 추출된 features를 입력받아 클러스터링만 수행 (API 호출 X)
        """
        # 1. 가중치 거리 행렬 계산
        dist_matrix = self._compute_matrix(photos, features)
        
        # 2. HDBSCAN
        try:
            clusterer = HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric='precomputed',
                cluster_selection_epsilon=self.eps, # 튜닝된 엡실론 사용
                cluster_selection_method='leaf'
            )
            labels = clusterer.fit_predict(dist_matrix)
        except Exception:
            # 실패 시 모두 -1(노이즈) 처리
            labels = np.full(len(photos), -1)
            
        return labels

    def _compute_matrix(self, photos, features):
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat, p.lon] for p in photos])
        
        # GPS 거리 계산 (반복문 최적화를 위해 단순화하거나 geod 유지)
        # 1000장이면 반복문이 50만번 돌기 때문에, 여기서 시간이 좀 걸림.
        # 최적화: pdist 등을 쓰면 좋지만, 가중치 로직 때문에 이중 루프 유지
        
        for i in range(n):
            for j in range(i + 1, n):
                _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                
                weight_factor = 1.0
                
                if features[i] is not None and features[j] is not None:
                    similarity = np.dot(features[i], features[j]) # features는 이미 norm 되어 있다고 가정
                    struct_dist = 1.0 - similarity
                    
                    # --- [튜닝 포인트: 동적 가중치 로직] ---
                    if struct_dist < self.strict_thresh:
                        weight_factor = self.w_merge # 예: 0.1
                    elif struct_dist > self.loose_thresh:
                        weight_factor = self.w_split # 예: 5.0
                    else:
                        # 선형 보간: strict와 loose 사이를 부드럽게 연결
                        # (복잡하면 단순 1.0 처리해도 됨)
                        slope = (self.w_split - self.w_merge) / (self.loose_thresh - self.strict_thresh)
                        weight_factor = self.w_merge + slope * (struct_dist - self.strict_thresh)

                final_dist = gps_dist * weight_factor
                
                if gps_dist > self.max_gps_tol:
                     # w_merge보다 조금이라도 크면 컷
                    if weight_factor > (self.w_merge + 0.1): 
                        final_dist = 1000.0

                dist_matrix[i][j] = dist_matrix[j][i] = final_dist
        
        return dist_matrix