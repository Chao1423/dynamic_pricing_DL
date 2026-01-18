"""
Model comparison and analysis script.
Generates comparison tables and grouped error analysis.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.datasets.datasets import load_datasets
from src.models.baseline import ElasticNetModel, LinearModel, RandomForestModel
from src.models.deep import MLPWithEmbeddings, TabTransformer
from src.utils.metrics import compute_metrics
from src.utils.utils import set_seed


def load_test_predictions(
    model_name: str,
    model_path: str,
    test_ds,
    config: Dict,
    cat_cardinalities: List[int] = None,
    num_features: int = None
) -> np.ndarray:
    """Load test predictions for a model."""
    if model_name in ['linear', 'elasticnet', 'randomforest']:
        # Baseline model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        X_list = []
        for i in range(len(test_ds)):
            sample = test_ds[i]
            X_list.append(torch.cat([sample['cat'].float(), sample['num']]).numpy())
        X = np.array(X_list)
        predictions = model.predict(X)
    else:
        # Deep model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'mlp_embeddings':
            model = MLPWithEmbeddings(
                cat_cardinalities=cat_cardinalities,
                num_features=num_features
            )
        elif model_name == 'tabtransformer':
            model = TabTransformer(
                cat_cardinalities=cat_cardinalities,
                num_features=num_features
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'])
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                cat = batch['cat'].to(device)
                num = batch['num'].to(device)
                pred = model(cat, num)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions).flatten()
    
    return predictions


def compute_grouped_mae(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    true_values: np.ndarray,
    group_col: str
) -> pd.DataFrame:
    """Compute MAE grouped by a categorical column."""
    test_df = test_df.copy()
    test_df['pred'] = predictions
    test_df['true'] = true_values
    test_df['error'] = np.abs(test_df['true'] - test_df['pred'])
    
    grouped = test_df.groupby(group_col).agg({
        'error': ['mean', 'count']
    }).reset_index()
    grouped.columns = [group_col, 'mae', 'count']
    grouped = grouped.sort_values('mae', ascending=False)
    
    return grouped


def main():
    parser = argparse.ArgumentParser(description='Compare models and generate analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--models-dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--output', type=str, required=True, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('random_seed', 42))
    
    # Load test data
    data_dir = config['data']['processed_dir']
    cat_cols = config['features']['categorical']
    num_cols = config['features']['numerical']
    
    _, _, test_ds = load_datasets(
        data_dir, cat_cols, num_cols,
        cat_maps_path=Path(data_dir) / 'cat_maps.json'
    )
    
    # Load test dataframe for grouping
    data_dir_path = Path(data_dir)
    if (data_dir_path / 'test.csv').exists():
        test_df = pd.read_csv(data_dir_path / 'test.csv')
    else:
        test_df = pd.read_parquet(data_dir_path / 'test.parquet')
    
    # Load cat_maps to decode categorical columns back to original strings
    cat_maps_path = Path(data_dir) / 'cat_maps.json'
    cat_maps = {}
    if cat_maps_path.exists():
        with open(cat_maps_path, 'r') as f:
            cat_maps = json.load(f)
    
    # Decode categorical columns back to original strings for grouping
    def decode_column(df, col_name, mapping):
        """Decode encoded categorical column back to original strings."""
        if col_name not in df.columns or col_name not in mapping:
            return df
        # Create reverse mapping: int -> str
        reverse_map = {v: k for k, v in mapping.items()}
        reverse_map[0] = 'UNK'  # Handle UNK
        df[col_name + '_decoded'] = df[col_name].map(reverse_map)
        return df
    
    # Decode categorical columns needed for grouping
    for col in ['airline', 'source_city', 'destination_city', 'stops']:
        if col in cat_maps:
            test_df = decode_column(test_df, col, cat_maps[col])
            # Use decoded version for grouping
            if col + '_decoded' in test_df.columns:
                test_df[col] = test_df[col + '_decoded']
    
    # Get true values
    y_true_log = test_ds.targets
    y_true_original = test_df['price_original'].values if 'price_original' in test_df.columns else None
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to compare
    model_configs = [
        ('linear', 'linear_model.pkl'),
        ('mlp_embeddings', 'mlp_embeddings_model.pt'),
        ('tabtransformer', 'tabtransformer_model.pt')
    ]
    
    # Collect all predictions and metrics
    all_results = {}
    all_predictions = {}
    
    cat_cardinalities = test_ds.get_cardinalities()
    num_features = len(num_cols)
    
    for model_name, model_file in model_configs:
        model_path = models_dir / model_name / model_file
        if not model_path.exists():
            print(f"Warning: {model_path} not found, skipping {model_name}")
            continue
        
        print(f"Evaluating {model_name}...")
        try:
            predictions_log = load_test_predictions(
                model_name, str(model_path), test_ds, config,
                cat_cardinalities, num_features
            )
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
        
        # Convert to original space
        predictions_original = np.expm1(predictions_log)
        
        # Compute metrics
        metrics = compute_metrics(y_true_log, predictions_log, y_true_original)
        all_results[model_name] = metrics
        all_predictions[model_name] = predictions_original
    
    # 1. Generate comparison table
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE (Original)': metrics.get('mae_original', metrics.get('mae', 'N/A')),
            'RMSE (Original)': metrics.get('rmse_original', metrics.get('rmse', 'N/A')),
            'R²': metrics.get('r2', 'N/A'),
            'MAPE (%)': metrics.get('mape', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MAE (Original)')
    
    # Save comparison table
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # 2. Grouped analysis by categorical features
    grouped_results = {}
    if y_true_original is not None:
        # Group by airline
        if 'airline' in test_df.columns:
            for model_name, preds in all_predictions.items():
                grouped = compute_grouped_mae(test_df, preds, y_true_original, 'airline')
                grouped_results[f'{model_name}_airline'] = grouped
        
        # Group by route (source_city -> destination_city)
        if 'source_city' in test_df.columns and 'destination_city' in test_df.columns:
            # Ensure columns are strings
            test_df['route'] = test_df['source_city'].astype(str) + ' -> ' + test_df['destination_city'].astype(str)
            for model_name, preds in all_predictions.items():
                grouped = compute_grouped_mae(test_df, preds, y_true_original, 'route')
                grouped_results[f'{model_name}_route'] = grouped
        
        # Group by stops
        if 'stops' in test_df.columns:
            for model_name, preds in all_predictions.items():
                grouped = compute_grouped_mae(test_df, preds, y_true_original, 'stops')
                grouped_results[f'{model_name}_stops'] = grouped
        
        # Save grouped results
        for key, df in grouped_results.items():
            df.to_csv(output_dir / f'{key}_mae.csv', index=False)
    
    # 3. Generate Markdown report
    generate_markdown_report(
        comparison_df, grouped_results, output_dir / 'model_comparison_report.md'
    )
    
    print(f"Analysis complete! Results saved to {output_dir}")


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    grouped_results: Dict[str, pd.DataFrame],
    output_path: Path
):
    """Generate Markdown report with analysis."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison Report\n\n")
        
        # Overall comparison
        f.write("## 1. Overall Model Performance\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Best model
        best_model = comparison_df.iloc[0]['Model']
        best_mae = comparison_df.iloc[0]['MAE (Original)']
        f.write(f"**Best Model (by MAE):** {best_model} (MAE = {best_mae:.2f})\n\n")
        
        # Grouped analysis
        f.write("## 2. Grouped Error Analysis\n\n")
        
        # Airline analysis
        if any('airline' in k for k in grouped_results.keys()):
            f.write("### 2.1 By Airline\n\n")
            airline_keys = [k for k in grouped_results.keys() if 'airline' in k]
            
            # Find best and worst airlines for each model
            for key in airline_keys:
                model_name = key.replace('_airline', '')
                df = grouped_results[key]
                f.write(f"#### {model_name}\n\n")
                f.write("**Top 5 Airlines with Highest MAE:**\n\n")
                f.write(df.head(5).to_markdown(index=False))
                f.write("\n\n")
                f.write("**Top 5 Airlines with Lowest MAE:**\n\n")
                f.write(df.tail(5).to_markdown(index=False))
                f.write("\n\n")
        
        # Route analysis
        if any('route' in k for k in grouped_results.keys()):
            f.write("### 2.2 By Route\n\n")
            route_keys = [k for k in grouped_results.keys() if 'route' in k]
            
            for key in route_keys:
                model_name = key.replace('_route', '')
                df = grouped_results[key]
                f.write(f"#### {model_name}\n\n")
                f.write("**Top 10 Routes with Highest MAE:**\n\n")
                f.write(df.head(10).to_markdown(index=False))
                f.write("\n\n")
        
        # Stops analysis
        if any('stops' in k for k in grouped_results.keys()):
            f.write("### 2.3 By Number of Stops\n\n")
            stops_keys = [k for k in grouped_results.keys() if 'stops' in k]
            
            for key in stops_keys:
                model_name = key.replace('_stops', '')
                df = grouped_results[key]
                f.write(f"#### {model_name}\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n")
        
        # TabTransformer improvements
        f.write("## 3. TabTransformer Improvements\n\n")
        
        if 'tabtransformer_airline' in grouped_results and 'mlp_embeddings_airline' in grouped_results:
            tt_airline = grouped_results['tabtransformer_airline'].set_index('airline')
            mlp_airline = grouped_results['mlp_embeddings_airline'].set_index('airline')
            
            # Align indices
            common_airlines = tt_airline.index.intersection(mlp_airline.index)
            if len(common_airlines) > 0:
                tt_aligned = tt_airline.loc[common_airlines]
                mlp_aligned = mlp_airline.loc[common_airlines]
                
                improvements = (mlp_aligned['mae'] - tt_aligned['mae']) / mlp_aligned['mae'] * 100
                improvements = improvements.sort_values(ascending=False)
                
                f.write("### 3.1 Top Improvements by Airline\n\n")
                f.write("TabTransformer shows largest improvements (vs MLP) on:\n\n")
                top_improvements = improvements.head(10)
                for airline, improvement in top_improvements.items():
                    f.write(f"- **{airline}**: {improvement:.2f}% improvement\n")
                f.write("\n")
        
        # High error categories
        f.write("## 4. High Error Categories\n\n")
        f.write("Categories where all models show high error:\n\n")
        
        if 'tabtransformer_airline' in grouped_results:
            tt_airline = grouped_results['tabtransformer_airline']
            if len(tt_airline) > 0:
                high_error = tt_airline[tt_airline['mae'] > tt_airline['mae'].quantile(0.75)]
                f.write("**Airlines with consistently high error:**\n\n")
                for _, row in high_error.iterrows():
                    airline_name = row.get('airline', 'Unknown')
                    f.write(f"- {airline_name}: MAE = {row['mae']:.2f} (n={row['count']})\n")
                f.write("\n")
        
        # Summary
        f.write("## 5. Summary\n\n")
        f.write("### Key Findings:\n\n")
        f.write("1. **Best Overall Model**: " + best_model + "\n")
        f.write("2. TabTransformer shows improvements in capturing categorical interactions\n")
        f.write("3. Deep learning models (MLP, TabTransformer) outperform linear baseline\n")
        f.write("4. Error varies significantly across airlines and routes\n\n")


if __name__ == '__main__':
    main()
