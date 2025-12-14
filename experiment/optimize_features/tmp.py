# class TunableHybridCluster:
#     def __init__(self, params: dict):
#         self.geod = Geod(ellps="WGS84")
#         self.params = params
        
#         # Common params
#         self.eps = params.get('eps', 8.0)
#         self.max_gps_tol = params.get('max_gps_tol', 40.0)
        
#         # Image Feature params
#         self.strict_thresh = params.get('strict_thresh', 0.15)
#         self.loose_thresh = params.get('loose_thresh', 0.35)
#         self.w_merge = params.get('w_merge', 0.1)
#         self.w_split = params.get('w_split', 5.0)
#         self.time_decay_slope = params.get('time_decay_slope', 0.05)
        
        # Time params (for GPS+Time mode)
        # w_time: meters per second (speed penalty) or simple linear weight?
        # Let's use it as: distance += w_time * abs(time_diff_minutes)
        # self.w_time = params.get('w_time', 0.0) 
        
        # HDBSCAN params
    #     self.min_cluster_size = params.get('min_cluster_size', 2)
    #     self.min_samples = params.get('min_samples', 2)

    # def run_clustering(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
    #     """
    #     Runs clustering on precomputed features.
    #     """
    #     if not photos:
    #         return np.array([])
            
    #     dist_matrix = self._compute_matrix(photos, features)
        
    #     try:
    #         clusterer = HDBSCAN(
    #             min_cluster_size=self.min_cluster_size,
    #             min_samples=self.min_samples,
    #             metric='precomputed',
    #             cluster_selection_epsilon=self.eps,
    #             cluster_selection_method='leaf'
    #         )
            # labels = clusterer.fit_predict(dist_matrix)
#         except Exception as e:
#             logger.error(f"HDBSCAN failed: {e}")
#             labels = np.full(len(photos), -1)
            
#         return labels
    
#     def _compute_matrix(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
#         n = len(photos)
#         dist_matrix = np.zeros((n, n))
        
#         # [중요] loose_thresh를 파라미터 튜닝에서 빼버리고 0.6으로 하드코딩 권장
#         # 작업자/공사전후 차이를 수용하기 위한 마지노선입니다.
#         HARD_LOOSE_THRESH = 0.6 
        
#         coords = np.array([[p.lat if p.lat else 0.0, p.lon if p.lon else 0.0] for p in photos])
        
#         for i in range(n):
#             for j in range(i + 1, n):
#                 # 1. GPS Distance
#                 _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
#                 if gps_dist < 3.0: gps_dist = 0.0
                
#                 final_dist = gps_dist
                
#                 # [Time Penalty 삭제 권장] 
#                 # 공사 전/후를 묶어야 하므로 시간 차이에 페널티를 주면 안 됩니다.
#                 # 굳이 쓴다면 아주 작게 (0.001) 유지.
                
#                 # 2. Visual Logic (Veto Power)
#                 if features[i] is not None and features[j] is not None:
#                     similarity = np.dot(features[i], features[j])
#                     visual_diff = 1.0 - similarity 
                    
#                     if visual_diff < 0.2:
#                         # [Zone 1] 완전 똑같음 -> 당겨줌
#                         final_dist *= 0.5
                        
#                     elif visual_diff > HARD_LOOSE_THRESH: 
#                         # [Zone 3] 명백히 다른 장소/각도 (0.6 이상)
#                         # GPS가 같아도(0m) 강제로 찢어놓음
#                         # w_split을 곱하는 대신 덧셈으로 확실하게 밀어냄
#                         final_dist += 20.0 
                        
#                     else:
#                         # [Zone 2] 작업자, 공사 전후, 계절 변화 (0.2 ~ 0.6)
#                         # ★ 여기가 핵심 ★
#                         # 이 구간은 건드리지 않고 GPS 거리 그대로 둠.
#                         # (Hybrid 로직이 점수를 깎아먹지 않게 방어)
#                         pass 

#                 dist_matrix[i][j] = dist_matrix[j][i] = final_dist
        
#         return dist_matrix

#     # def _compute_matrix(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
#     #     n = len(photos)
#     #     dist_matrix = np.zeros((n, n))
        
#     #     coords = np.array([[p.lat if p.lat else 0.0, p.lon if p.lon else 0.0] for p in photos])
#     #     timestamps = np.array([p.timestamp if p.timestamp else 0.0 for p in photos])
        
#     #     for i in range(n):
#     #         for j in range(i + 1, n):
#     #             # 1. GPS Distance
#     #             _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
#     #             if gps_dist < 5.0: gps_dist = 0.0 # 노이즈 제거
                
#     #             final_dist = gps_dist
                
#     #             # 2. Time Distance
#     #             if self.w_time > 0:
#     #                 time_diff = abs(timestamps[i] - timestamps[j]) / 60.0 # Difference in minutes
#     #                 final_dist += self.w_time * time_diff

#     #             # 3. Image Feature Weighting OR pure GPS-Time
#     #             if features[i] is not None and features[j] is not None:
#     #                 # Image features are present, apply weighting
#     #                 # Assumes normalized features
#     #                 similarity = np.dot(features[i], features[j])
#     #                 struct_dist = 1.0 - similarity
                    
#     #                 weight_factor = 1.0
#     #                 if struct_dist < self.strict_thresh:
#     #                     weight_factor = self.w_merge
#     #                 elif struct_dist > self.loose_thresh:
#     #                     weight_factor = self.w_split
#     #                 else:
#     #                     # Linear interpolation
#     #                     slope = (self.w_split - self.w_merge) / (self.loose_thresh - self.strict_thresh)
#     #                     weight_factor = self.w_merge + slope * (struct_dist - self.strict_thresh)

#     #                 final_dist *= weight_factor
#     #             else:
#     #                 # No image features. weight_factor is implicitly 1.0.
#     #                 # The distance is purely based on GPS and Time.
#     #                 pass
                
#     #             # 4. Hard Constraints
#     #             # If GPS distance is too large, enforce separation unless image features strongly suggest merge.
#     #             if gps_dist > self.max_gps_tol:
#     #                 # When image features are present, weight_factor can be < 1.0,
#     #                 # potentially overriding gps_dist > max_gps_tol.
#     #                 # When no image features, weight_factor is effectively 1.0.
#     #                 # This check ensures that points far apart by GPS are separated
#     #                 # unless image features provide strong evidence to merge.
                    
#     #                 # If we have strong image similarity (low weight factor, e.g., <0.15)
#     #                 # we might "forgive" GPS distance being over tolerance.
#     #                 # Otherwise, if GPS distance is beyond tolerance, we separate them.
#     #                 current_weight_factor = 1.0 # Default if no image features
#     #                 if features[i] is not None and features[j] is not None:
#     #                      similarity = np.dot(features[i], features[j])
#     #                      struct_dist = 1.0 - similarity
#     #                      if struct_dist < self.strict_thresh:
#     #                          current_weight_factor = self.w_merge
#     #                      elif struct_dist > self.loose_thresh:
#     #                          current_weight_factor = self.w_split
#     #                      else:
#     #                          slope = (self.w_split - self.w_merge) / (self.loose_thresh - self.strict_thresh)
#     #                          current_weight_factor = self.w_merge + slope * (struct_dist - self.strict_thresh)

#     #                 if current_weight_factor > (self.w_merge + 0.1): # Strong separation if weight factor is not very low
#     #                     final_dist = 1000.0

#     #             dist_matrix[i][j] = dist_matrix[j][i] = final_dist
        
#     #     return dist_matrix