import pickle

import optuna
from clusters import TunableHybridCluster
from dataset import photos
from sklearn.metrics import adjusted_rand_score


def objective(trial):
    # 1. 데이터 로드 (캐시된 데이터)
    with open("dataset_cache.pkl", "rb") as f:
        data = pickle.load(f)
    photos = data['photos']
    features = data['features']
    
    # 정답 라벨 추출 (Ground Truth)
    # PhotoMeta 객체에 'label_id' 같은 속성이 있다고 가정
    true_labels = [p.label_id for p in photos] 
    
    # 2. 하이퍼파라미터 탐색 범위 설정
    params = {
        # 구조적 유사도 임계값 (Vertex AI 기준)
        "strict_thresh": trial.suggest_float("strict_thresh", 0.10, 0.25),
        "loose_thresh": trial.suggest_float("loose_thresh", 0.30, 0.50),
        
        # HDBSCAN Epsilon (Weighted Meter)
        "eps": trial.suggest_float("eps", 1.0, 5.0),
        
        # GPS 허용 오차
        "max_gps_tol": trial.suggest_float("max_gps_tol", 30.0, 60.0),
        
        # 가중치 강도
        "w_merge": trial.suggest_float("w_merge", 0.05, 0.3), # 작을수록 강한 결합
        "w_split": trial.suggest_float("w_split", 3.0, 8.0),  # 클수록 강한 분리
    }
    
    # 논리적 제약 조건 (Strict < Loose 여야 함)
    if params["strict_thresh"] >= params["loose_thresh"]:
        # 말이 안 되는 조합은 가지치기(Pruning)
        raise optuna.TrialPruned()

    # 3. 클러스터링 실행
    clusterer = TunableHybridCluster(params)
    pred_labels = clusterer.run_clustering(photos, features)
    
    # 4. 성능 평가 (Adjusted Rand Index)
    # ARI는 1.0에 가까울수록 정답과 완벽히 일치함 (0.0은 무작위)
    score = adjusted_rand_score(true_labels, pred_labels)
    
    return score

# --- [Main Execution] ---

if __name__ == "__main__":
    # 1. 스터디 생성 (Maximize ARI)
    study = optuna.create_study(direction="maximize")
    
    # 2. 최적화 실행 (n_trials=100번 시도)
    print("Start optimization...")
    study.optimize(objective, n_trials=100)
    
    # 3. 결과 출력
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)
    
    # 시각화 (Jupyter 환경이라면)
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()