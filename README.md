# Airline Ticket Price Prediction

A reproducible machine learning project for predicting airline ticket prices using baseline and deep learning models.

## Project Structure

```
.
├── src/
│   ├── preprocess/          # Data preprocessing module
│   │   └── preprocess.py    # Main preprocessing pipeline
│   ├── datasets/            # Dataset classes
│   │   └── datasets.py      # PyTorch Dataset implementation
│   ├── models/              # Model definitions
│   │   ├── baseline.py     # Linear, ElasticNet, RandomForest
│   │   └── deep.py         # MLP+Embeddings, TabTransformer
│   ├── train/              # Training scripts
│   │   └── train.py        # Model training
│   ├── eval/               # Evaluation scripts
│   │   └── eval.py         # Model evaluation
│   └── utils/              # Utility functions
│       ├── metrics.py      # Evaluation metrics
│       └── utils.py        # Helper functions
├── configs/                # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Data directory
│   └── processed/          # Processed data outputs
├── models/                 # Saved models
├── reports/                # Experiment reports
└── notebooks/              # Jupyter notebooks

```

## File Responsibilities

### Preprocessing (`src/preprocess/preprocess.py`)
- **parse_duration()**: Converts duration strings/numeric to minutes
- **extract_date_features()**: Extracts day, month, weekday, season from dates
- **create_days_left()**: Creates days_left feature if missing
- **encode_categorical()**: Encodes categoricals with UNK support
- **compute_iqr_thresholds()**: Computes IQR thresholds on train data only
- **remove_outliers()**: Removes outliers using pre-computed thresholds
- **preprocess_data()**: Main preprocessing pipeline
- **main()**: CLI entry point with argparse

### Datasets (`src/datasets/datasets.py`)
- **AirlineDataset**: PyTorch Dataset class
  - Handles categorical and numerical features separately
  - Returns dict with 'cat', 'num', 'target' keys
  - Provides cardinality information for embeddings
- **load_datasets()**: Loads train/val/test datasets

### Models

#### Baseline (`src/models/baseline.py`)
- **LinearModel**: Linear regression
- **ElasticNetModel**: ElasticNet regression
- **RandomForestModel**: Random Forest regression

#### Deep (`src/models/deep.py`)
- **MLPWithEmbeddings**: MLP with embedding layers for categoricals
- **TabTransformer**: Transformer-based model for tabular data

### Training (`src/train/train.py`)
- **train_baseline()**: Trains baseline models
- **train_deep()**: Trains deep learning models
- **prepare_baseline_data()**: Prepares data for baseline models
- **main()**: CLI entry point

### Evaluation (`src/eval/eval.py`)
- **evaluate_baseline()**: Evaluates baseline models
- **evaluate_deep()**: Evaluates deep learning models
- **main()**: CLI entry point

### Utils
- **metrics.py**: Regression metrics (RMSE, MAE, R2, etc.)
- **utils.py**: Seed setting for reproducibility

## Usage

### 1. Preprocessing

```bash
python -m src.preprocess.preprocess \
    --input Clean_Dataset.csv \
    --output data/processed \
    --cat-cols airline source_city destination_city departure_time arrival_time stops class \
    --num-cols duration_minutes days_left \
    --random-state 42
```

### 2. Training

```bash
# Train baseline model
python -m src.train.train \
    --config configs/config.yaml \
    --model linear \
    --output models/linear

# Train deep model
python -m src.train.train \
    --config configs/config.yaml \
    --model mlp_embeddings \
    --output models/mlp_embeddings
```

### 3. Evaluation

```bash
python -m src.eval.eval \
    --config configs/config.yaml \
    --model linear \
    --model-path models/linear/linear_model.pkl \
    --output reports/linear
```

## Configuration

Edit `configs/config.yaml` to modify:
- Feature lists
- Data paths
- Model hyperparameters
- Training settings
- Split ratios

## Key Features

- **Data leakage prevention**: All preprocessing (scaling, IQR) fit only on train data
- **Reproducibility**: Random seeds controlled throughout
- **UNK handling**: Categorical encoders support unseen categories
- **Flexible duration parsing**: Handles multiple duration formats
- **Comprehensive outputs**: Saves processed data, mappings, scalers, thresholds

## Installation

```bash
pip install -r requirements.txt
```
