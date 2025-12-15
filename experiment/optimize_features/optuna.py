import argparse
import asyncio
import glob
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from pyproj import Geod
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)

# Add project root and app directory to sys.path to import app modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
app_dir = project_root / "app"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from experiment.optimize_features.clusters import TunableHybridCluster
from experiment.optimize_features.dataset import load_dataset, prepare_dataset
from experiment.optimize_features.extractors import get_extractor
from experiment.optimize_features.masking import RoboflowMasker

try:
    from app.common.models import PhotoMeta
    from app.services.metadata_extractor import MetadataExtractor
except ImportError:
    pass

logger = logging.getLogger(__name__)


def load_ground_truth(sql_dump_paths: list[str]) -> dict[str, str]:
    """
    Parses multiple SQL dump files to create a combined mapping from original_filename to cluster_id.
    """
    mapping = {}
    for sql_path in sql_dump_paths:
        if not os.path.exists(sql_path):
            logger.warning(f"SQL dump not found at {sql_path}. Skipping.")
            continue
        
        try:
            with open(sql_path, 'r', encoding='utf-8') as f:
                in_copy_block = False
                for line in f:
                    line = line.strip()
                    if line.startswith("COPY public.photos"):
                        in_copy_block = True
                        continue
                    if line == r"\\.":
                        in_copy_block = False
                        continue

                    if in_copy_block:
                        # format: id job_id cluster_id original_filename ...
                        parts = line.split('\t')
                        if len(parts) < 4:
                            continue
                        
                        # job_id = parts[1]
                        cluster_id = parts[2]
                        original_filename = parts[3]
                        
                        if cluster_id != r'\N':
                            mapping[original_filename] = cluster_id
        except Exception as e:
            logger.error(f"Error reading SQL dump {sql_path}: {e}")
    
    return mapping


async def generate_dataset_from_paths(
    parent_dir: str, 
    sql_dump_paths: list[str], 
    output_path: str, 
    extractor_name: str, 
    masking: bool,
    mode: str, 
) -> str:
    """
    Generates a dataset pickle file from a parent directory (recursive search) and SQL dumps.
    """
    # 1. Load Ground Truth
    cluster_mapping = {}
    if sql_dump_paths:
        print(f"Loading ground truth from {len(sql_dump_paths)} files...")
        cluster_mapping = load_ground_truth(sql_dump_paths)
        print(f"Loaded {len(cluster_mapping)} labels.")

    # 2. Extract Metadata
    extractor_meta = MetadataExtractor()
    photos: list[PhotoMeta] = []
    
    print(f"Recursively searching for images in {parent_dir}...")
    if not os.path.isdir(parent_dir):
        print(f"Error: {parent_dir} is not a directory.")
        return

    # Support common image extensions
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(parent_dir, "**", ext), recursive=True))
        
    print(f"Found {len(files)} images.")
    
    for f in files:
        meta = await extractor_meta.extract(f)
        # Filter for valid GPS if needed (usually required for this clustering)
        if meta and meta.lat is not None and meta.lon is not None:
            # Inject label_id
            if meta.original_name in cluster_mapping:
                setattr(meta, 'label_id', cluster_mapping[meta.original_name])
            else:
                # Optional: Mark as noise or unknown? 
                # For optimization, we usually only want labeled data.
                # But we'll keep it, optimization loop filters missing labels.
                pass
            photos.append(meta)

    # correct_outliers_by_speed(photos)
    # adjust_gps_inaccuracy(photos)
    print(f"Collected {len(photos)} photos with valid GPS.")

    # 3. Extract Features
    if mode == "gps_time":
        feature_extractor = get_extractor("null")
    else:
        feature_extractor = get_extractor(extractor_name)

    masker = None
    if masking:
        masker = RoboflowMasker()
    prepare_dataset(feature_extractor, photos, output_path, masker=masker)
    return output_path


def correct_outliers_by_speed(photos): 
    geod = Geod(ellps="WGS84")

    timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
    timed_photos.sort(key=lambda x: x.timestamp)
    max_speed_mps = 5.0 
    for i in range(1, len(timed_photos)):
        prev = timed_photos[i-1]
        curr = timed_photos[i]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0: continue
        _, _, dist = geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
        if (dist / dt) > max_speed_mps:
            curr.lat = prev.lat
            curr.lon = prev.lon
            if prev.alt is not None: curr.alt = prev.alt


def adjust_gps_inaccuracy(photos): 
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


def get_params_for_mode(trial, mode):
    """
    Define search space based on the selected mode.
    """
    params = {
        # Common
        "eps": trial.suggest_float("eps", 5.0, 10.0),
        "max_gps_tol": trial.suggest_float("max_gps_tol", 30.0, 60.0),
        
        # HDBSCAN Structural Params
        # Photos per cluster: Mostly 3-5, sometimes 2, sometimes >5.
        # To prevent over-segmentation, we might want slightly higher min_cluster_size 
        # but since 2 is possible, we must allow 2. 
        # min_samples controls how conservative the clustering is (higher = more noise).
        "min_cluster_size": trial.suggest_int("min_cluster_size", 2, 4),
        "min_samples": trial.suggest_int("min_samples", 1, 3),
    }

    if mode == "gps_time":
        # GPS + Time only
        params["w_time"] = trial.suggest_float("w_time", 0.0, 1.0)
        
    else:
        # Image Feature Modes (vertex, mobilenet)
        params["w_time"] = 0.0 
        
        params["strict_thresh"] = trial.suggest_float("strict_thresh", 0.10, 0.25)
        params["loose_thresh"] = trial.suggest_float("loose_thresh", 0.30, 0.50)
        params["w_merge"] = trial.suggest_float("w_merge", 0.05, 0.3)
        params["w_split"] = trial.suggest_float("w_split", 3.0, 8.0)

        # Logical constraints
        if params["strict_thresh"] >= params["loose_thresh"]:
            raise optuna.TrialPruned()

    return params


def objective(trial, photos, features, true_labels, mode):
    params = get_params_for_mode(trial, mode)
    
    clusterer = TunableHybridCluster(params)
    pred_labels = clusterer.run_clustering(photos, features)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    
    # Use V-Measure (Harmonic Mean of Homogeneity and Completeness)
    # beta=1.0 (default) weighs them equally, striving to maximize BOTH.
    v_measure = v_measure_score(true_labels, pred_labels, beta=1.5)
    
    # Store metrics in user attributes for analysis later
    trial.set_user_attr("ari", ari)
    trial.set_user_attr("homogeneity", homogeneity)
    trial.set_user_attr("completeness", completeness)
    trial.set_user_attr("v_measure", v_measure)
    
    return homogeneity


def run_optimization(dataset_path: str, n_trials: int, mode: str):
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)
    photos = data['photos']
    features = data['features']
    
    if mode == "gps_time":
        print("Mode is gps_time. Ignoring image features in dataset.")
        features = [None] * len(photos)
    
    # Extract labels
    true_labels = []
    valid_indices = []
    
    for i, p in enumerate(photos):
        label = None
        if hasattr(p, 'label_id'):
            label = p.label_id
        elif isinstance(p, dict) and 'label_id' in p:
            label = p['label_id']
        
        if label is not None:
            true_labels.append(label)
            valid_indices.append(i)
    
    if not true_labels:
        print("Error: Could not find any ground truth labels.")
        return

    # Filter photos and features to only those with labels
    if len(true_labels) < len(photos):
        print(f"Filtering dataset: Using {len(true_labels)} labeled samples out of {len(photos)}.")
        photos = [photos[i] for i in valid_indices]
        features = [features[i] for i in valid_indices]

    print(f"Starting optimization (Mode: {mode}) with {len(photos)} samples and {n_trials} trials...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, photos, features, true_labels, mode), n_trials=n_trials)
    
    print("Best Score:", study.best_value)
    print("Best params:", study.best_params)
    
    # Print detailed metrics for the best trial
    best_trial = study.best_trial
    print("\n--- Detailed Metrics for Best Trial ---")
    print(f"ARI:         {best_trial.user_attrs.get('ari', 'N/A'):.4f}")
    print(f"Homogeneity: {best_trial.user_attrs.get('homogeneity', 'N/A'):.4f}")
    print(f"Completeness: {best_trial.user_attrs.get('completeness', 'N/A'):.4f}")
    print(f"V-Measure:   {best_trial.user_attrs.get('v_measure', 'N/A'):.4f}")
    print("---------------------------------------")
    
    # Save best params to json
    save_path = os.path.join(os.path.dirname(dataset_path), "best_params.json")
    import json
    with open(save_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best parameters saved to {save_path}")

    return study


def load_best_params(dataset_path: str = "./experiment/optimize_features/features/dataset_cache.pkl") -> dict:
    save_path = os.path.join(os.path.dirname(dataset_path), "best_params.json")
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cache", type=bool, default=False, help="Diretory path to dataset pickle.")
    parser.add_argument("--dataset_dir", type=str, default="./experiment/optimize_features/features/", help="Diretory path to dataset pickle.")
    parser.add_argument("--dataset_name", type=str, help="Name to dataset pickle.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials.")
    parser.add_argument("--masking", type=bool, default=False, help="Masking before feature extraction.")
    parser.add_argument("--mode", type=str, default="mobilenet", choices=["mobilenet", "vertex", "gps_time", "cosplace"], 
                        help="Optimization mode.")
    
    # Generation args
    parser.add_argument("--generate-from", type=str, default="./assets/labeled", help="Parent directory to recursively search for images.")
    parser.add_argument("--sql-dump", nargs='+', help="list of paths to SQL dumps for ground truth labels.")
    
    args = parser.parse_args()

    if args.use_cache:
        dataset_path = os.path.join(args.dataset_dir, args.dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.\nUse --dataset_name for use_cache in {args.dataset_dir}. Or Use --generate-from to create one.")
            exit(1)
    else:
        if args.dataset_name:
            dataset_name = args.dataset_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"dataset_{args.mode}_{'masking' if args.masking else 'nomask'}_{timestamp}.pkl"

        sql_dump = args.sql_dump
        if not sql_dump:
            sql_dump = [
                "./assets/dataset_sqldump/photo1.sql", 
                "./assets/dataset_sqldump/photo2.sql", 
                "./assets/dataset_sqldump/photo0.sql",
            ]
            print(f"Warning: --sql-dump not provided. Use default label with {sql_dump}.\n\
                  If they don't include your lable, Dataset will have no labels (useful only for inference tests, not optimization).")

        print(args.dataset_dir, dataset_name)
        output_path = os.path.join(args.dataset_dir, dataset_name)
        print(f"Generating dataset at {output_path} from images in {args.generate_from}...")

        # Use asyncio to run the generator
        dataset_path = asyncio.run(generate_dataset_from_paths(
            parent_dir=args.generate_from,
            sql_dump_paths=sql_dump,
            output_path=output_path,
            extractor_name=args.mode,
            masking=args.masking,
            mode=args.mode
        ))

    run_optimization(dataset_path, args.trials, args.mode)
