import logging
import math
from typing import List, Optional

import numpy as np
from pyproj import Geod
from sklearn.cluster import KMeans

# Try to import HDBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    import hdbscan as HDBSCAN

# App imports (가정)
from app.common.models import PhotoMeta

logger = logging.getLogger(__name__)

class TunableHybridCluster:
    def __init__(self, params: dict):
        self.geod = Geod(ellps="WGS84")
        self.params = params
        
        # 1. GPS Clustering Params (1단계)
        self.gps_eps = 5.0  # 요청하신 대로 10m 고정 (또는 params에서 가져오기)
        self.max_gps_tol = params.get('max_gps_tol', 40.0)
        
        # 2. Visual Split Params (2단계)
        # 이미지로 나눌 때 "다르다"고 판단할 기준 (0.4 ~ 0.6 추천)
        self.visual_split_thresh = params.get('loose_thresh', 0.5) 
        self.split_min_size = 5  # 이 숫자보다 클 때만 이미지 분할 시도
        self.min_cluster_size = params.get('min_cluster_size', 2)
        self.min_samples = params.get('min_samples', 2)
        
        # 3. Size Enforcement (3단계)
        self.max_cluster_size = params.get('max_cluster_size', 4)

    def run_clustering(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
        """
        2-Step Clustering with Size Limit:
        Step 1. GPS Distance (eps=10)로 1차 그룹핑
        Step 2. 그룹 내 사진 수가 split_min_size 넘으면 Visual Distance로 2차 분할 (HDBSCAN)
        Step 3. 여전히 max_cluster_size(5)를 넘는 클러스터는 K-Means로 강제 분할
        """
        if not photos:
            return np.array([])
            
        n_samples = len(photos)
        
        # --- [Step 1] GPS 기반 1차 클러스터링 ---
        gps_matrix = self._compute_gps_matrix(photos)
        
        try:
            gps_clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='precomputed',
                cluster_selection_epsilon=self.gps_eps, # eps = 8.0
                cluster_selection_method='eom',
                allow_single_cluster=True
            )
            # 1차 라벨 생성 (예: 0, 0, 0, 1, 1, -1, 2, 2...)
            labels = gps_clusterer.fit_predict(gps_matrix)
        except Exception as e:
            logger.error(f"Step 1 GPS HDBSCAN failed: {e}")
            return np.full(n_samples, -1)

        # --- [Step 2] 거대 클러스터 대상 이미지 기반 재분할 (HDBSCAN) ---
        
        # 새로운 라벨을 부여하기 위해 현재 최대 라벨 번호 확인
        max_label = labels.max()
        next_label_id = max_label + 1
        
        # 존재하는 각 클러스터 ID에 대해 반복 (노이즈 -1 제외)
        unique_labels = set(labels)
        if -1 in unique_labels: 
            unique_labels.remove(-1)
            
        for cluster_id in unique_labels:
            # 해당 클러스터에 속한 사진의 인덱스 추출
            indices = np.where(labels == cluster_id)[0]
            
            # 조건: 사진 수가 분할 최소 크기보다 큰가?
            if len(indices) > self.split_min_size:
                
                # 해당 그룹의 Feature만 추출
                sub_features = [features[i] for i in indices]
                
                # Feature가 하나라도 없으면 분할 불가 (Skip)
                if any(f is None for f in sub_features):
                    continue
                
                # 서브 그룹용 Visual Distance Matrix 계산
                visual_matrix = self._compute_visual_matrix(sub_features)
                
                try:
                    # 2차 클러스터링 (이미지 유사도 기반)
                    sub_clusterer = HDBSCAN(
                        min_cluster_size=2,
                        min_samples=2,
                        metric='precomputed',
                        cluster_selection_epsilon=self.visual_split_thresh,
                        allow_single_cluster=False 
                    )
                    sub_labels = sub_clusterer.fit_predict(visual_matrix)
                    
                    found_sub_clusters = set(sub_labels)
                    
                    for sub_id in found_sub_clusters:
                        sub_indices = indices[sub_labels == sub_id]
                        
                        # 기존 cluster_id를 덮어씌움 (모두 새 ID 부여하여 충돌 방지)
                        labels[sub_indices] = next_label_id
                        next_label_id += 1
                        
                except Exception as e:
                    logger.warning(f"Step 2 Visual Split failed for cluster {cluster_id}: {e}")
                    # 실패 시 기존 라벨 유지

        # --- [Step 3] Force Split if cluster size > max_cluster_size (K-Means) ---
        
        # Step 2까지 거친 labels에 대해 다시 검사
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        # list로 복사해서 순회 (labels 배열이 바뀌므로)
        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            n_members = len(indices)
            
            if n_members > self.max_cluster_size:
                # 쪼개야 할 개수 계산 (올림)
                n_splits = math.ceil(n_members / self.max_cluster_size)
                
                # Feature 추출
                sub_features = [features[i] for i in indices]
                
                # Feature가 하나라도 없으면 분할 불가
                if any(f is None for f in sub_features):
                    continue 
                
                try:
                    # K-Means로 강제 분할
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(sub_features)
                    
                    # 서브 라벨 매핑 (0번부터 n_splits-1번까지)
                    for k in range(n_splits):
                        sub_indices = indices[sub_labels == k]
                        # 모두에게 새로운 ID 부여 (기존 ID 혼동 방지)
                        labels[sub_indices] = next_label_id
                        next_label_id += 1
                            
                except Exception as e:
                    logger.warning(f"Step 3 K-Means force split failed for cluster {cluster_id}: {e}")

        return labels

    def _compute_gps_matrix(self, photos: List[PhotoMeta]) -> np.ndarray:
        """순수 GPS 거리 매트릭스 계산"""
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat if p.lat else 0.0, p.lon if p.lon else 0.0] for p in photos])
        
        for i in range(n):
            for j in range(i + 1, n):
                _, _, dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix

    def _compute_visual_matrix(self, features: List[np.ndarray]) -> np.ndarray:
        """순수 Visual Distance 매트릭스 계산 (Cosine Distance)"""
        n = len(features)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine Similarity: -1 ~ 1 (1에 가까울수록 유사)
                similarity = np.dot(features[i], features[j])
                
                # Distance: 0 ~ 2 (0에 가까울수록 유사)
                # 보통 0.0 ~ 1.0 사이 (유사도 양수 가정 시)
                dist = 1.0 - similarity
                
                # 음수 방지 및 범위 보정
                if dist < 0: dist = 0.0
                
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix