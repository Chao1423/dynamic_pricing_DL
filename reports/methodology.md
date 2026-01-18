# Methodology

## Data Processing

- **Input**: `Clean_Dataset.csv`.
- **Target transformation**: `price` is transformed to `log1p(price)` for modeling stability, while `price_original` is preserved for evaluation in original scale.
- **Duration parsing**: `duration` is converted to minutes and stored as `duration_minutes`. The parser supports decimal hours (e.g., `2.5`) and string formats (e.g., `2h 50m`).
- **Days left**: `days_left` is used as a continuous variable. If missing, it is constructed from `departure_date - booking_date` when available; otherwise preprocessing fails with an explicit error.
- **Categorical encoding**: categorical features are mapped to integer indices with an `UNK` bucket for unseen categories.
- **Scaling**: numerical features are standardized using `StandardScaler`, fit on train only, then applied to val/test.
- **Outlier filtering**: IQR-based filtering (1.5x) is computed on the train split only, and the same thresholds are applied to val/test.
- **Split**: 80/20 train/test, then 12.5% of train used as validation.
- **Artifacts**: processed CSV/Parquet, `cat_maps.json`, `scaler.pkl`, and `iqr_thresholds.json` are saved.

## Baseline: Linear Regression

- **Features**: concatenation of encoded categorical indices and standardized numerical features.
- **Model**: `LinearRegression`.
- **Loss/Optimization**: solved by the closed-form linear regression solver in scikit-learn.
- **Evaluation**: MAE, RMSE, R², MAPE (original scale) computed on test set.

## Deep Model: MLP + Embeddings

- **Categorical features**: each categorical column has its own embedding layer.
- **Numerical features**: concatenated directly with embeddings.
- **Architecture**: 3-layer MLP with BatchNorm and Dropout.
- **Loss**: MAE (`L1Loss`).
- **Optimization**: Adam optimizer, early stopping based on validation MAE.
- **Outputs**: predictions in log space, converted to original scale for final metrics.

## Deep Model: TabTransformer

- **Categorical features**: embedded and passed through a Transformer Encoder to learn inter-feature interactions.
- **Numerical features**: projected to the same embedding dimension.
- **Fusion**: pooled categorical representation concatenated with projected numerical features.
- **Head**: MLP with BatchNorm and Dropout, outputs a single regression value.
- **Loss/Training**: same training and evaluation pipeline as MLP (MAE loss, early stopping).
- **Hyperparameters**: configurable `num_heads` and `num_layers` in YAML.

## Evaluation Protocol

- **Primary metric**: MAE on original price scale.
- **Additional metrics**: RMSE, R², MAPE.
- **Grouped analysis**: MAE computed by `airline`, `route`, and `stops` for error diagnostics and fairness comparison across models.
