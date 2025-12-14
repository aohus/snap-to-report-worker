## Directory Structure

```
experiment/optimize_features/
├── clusters.py         # Clustering logic
  (TunableHybridCluster)
├── dataset.py          # Dataset preparation & loading
├── obtuna.py           # Optimization entry point
└── extractors/         # [NEW] Feature extractors package
    ├── __init__.py     # Factory & exports
    ├── base.py         # Abstract Base Class
    ├── mobilenet.py    # MobileNet implementation
    └── vertex.py       # Vertex AI implementation
```

## Usage Example

1. Generate Dataset (if you have the ground truth loading logic implemented):

```python
# Need to implement loading logic in dataset.py or provide   source
python experiment/optimize_features/obtuna.py --generate-from "data.csv" --extractor mobilenet
```

2. Run Optimization:

```python
python experiment/optimize_features/obtuna.py --dataset dataset_cache.pkl --trials 100
```
