"""
Data preprocessing module for airline ticket price prediction.
Handles feature engineering, encoding, scaling, and outlier removal.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_duration(duration: str) -> float:
    """
    Parse duration string to minutes.
    Supports formats: '2h 50m', '2.5' (decimal hours), '150' (minutes).
    
    Args:
        duration: Duration string or numeric value
        
    Returns:
        Duration in minutes (float)
    """
    if pd.isna(duration):
        return np.nan
    
    # If already numeric (decimal hours)
    if isinstance(duration, (int, float)):
        return float(duration) * 60
    
    s = str(duration).lower().strip()
    
    # Try parsing as 'Xh Ym' format
    hours = 0
    minutes = 0
    
    if "h" in s:
        parts = s.split("h")
        hours = float(parts[0].strip())
        if len(parts) > 1 and "m" in parts[1]:
            min_part = parts[1].split("m")[0].strip()
            if min_part:
                minutes = float(min_part)
    elif "m" in s:
        # Only minutes
        minutes = float(s.split("m")[0].strip())
    else:
        # Try as decimal hours
        try:
            return float(s) * 60
        except ValueError:
            return np.nan
    
    return hours * 60 + minutes


def extract_date_features(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Extract date features: day, month, weekday, season.
    If date_col not exists, try to infer from other columns.
    
    Args:
        df: Input dataframe
        date_col: Name of date column (optional)
        
    Returns:
        DataFrame with added date features
    """
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df['day'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['weekday'] = df[date_col].dt.dayofweek
        df['season'] = df[date_col].dt.month % 12 // 3 + 1
    # If no date column, skip (data may not have it)
    
    return df


def create_days_left(df: pd.DataFrame, 
                    departure_col: str = None,
                    booking_col: str = None) -> pd.DataFrame:
    """
    Create days_left feature if not exists.
    
    Args:
        df: Input dataframe
        departure_col: Departure date column name
        booking_col: Booking date column name
        
    Returns:
        DataFrame with days_left column
    """
    if 'days_left' in df.columns:
        return df
    
    if departure_col and booking_col:
        if departure_col in df.columns and booking_col in df.columns:
            df[departure_col] = pd.to_datetime(df[departure_col])
            df[booking_col] = pd.to_datetime(df[booking_col])
            df['days_left'] = (df[departure_col] - df[booking_col]).dt.days
            return df
    
    raise ValueError(
        "days_left column not found and cannot be constructed. "
        "Please provide departure_col and booking_col, or ensure days_left exists."
    )


def encode_categorical(df: pd.DataFrame, 
                      cat_cols: List[str],
                      cat_maps: Dict[str, Dict[str, int]] = None,
                      fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Encode categorical variables to indices with UNK support.
    
    Args:
        df: Input dataframe
        cat_cols: List of categorical column names
        cat_maps: Pre-computed category mappings (for inference)
        fit: Whether to fit new mappings (True) or use existing (False)
        
    Returns:
        Tuple of (encoded_df, cat_maps)
    """
    if cat_maps is None:
        cat_maps = {}
    
    df_encoded = df.copy()
    
    for col in cat_cols:
        if col not in df.columns:
            continue
            
        if fit:
            # Fit: create mapping with UNK=0
            unique_vals = df[col].astype(str).unique()
            # Map: 0=UNK, 1,2,3... for actual categories
            mapping = {'UNK': 0}
            for idx, val in enumerate(sorted(unique_vals), start=1):
                mapping[val] = idx
            cat_maps[col] = mapping
        else:
            # Transform: use existing mapping
            if col not in cat_maps:
                raise ValueError(f"Mapping for {col} not found in cat_maps")
            mapping = cat_maps[col]
        
        # Apply mapping, UNK for unseen values
        df_encoded[col] = df[col].astype(str).map(
            lambda x: mapping.get(x, 0)  # 0 is UNK
        )
    
    return df_encoded, cat_maps


def compute_iqr_thresholds(df: pd.DataFrame, 
                          num_cols: List[str],
                          factor: float = 1.5) -> Dict[str, Dict[str, float]]:
    """
    Compute IQR thresholds for outlier detection (only on train data).
    
    Args:
        df: Training dataframe
        num_cols: List of numerical column names
        factor: IQR multiplier (default 1.5)
        
    Returns:
        Dictionary of thresholds per column
    """
    thresholds = {}
    
    for col in num_cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        thresholds[col] = {
            'lower': Q1 - factor * IQR,
            'upper': Q3 + factor * IQR
        }
    
    return thresholds


def remove_outliers(df: pd.DataFrame,
                   num_cols: List[str],
                   thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Remove outliers using pre-computed IQR thresholds.
    
    Args:
        df: Input dataframe
        num_cols: List of numerical column names
        thresholds: Pre-computed thresholds
        
    Returns:
        Filtered dataframe
    """
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in num_cols:
        if col not in thresholds:
            continue
        thresh = thresholds[col]
        mask &= (df[col] >= thresh['lower']) & (df[col] <= thresh['upper'])
    
    return df[mask].reset_index(drop=True)


def preprocess_data(
    input_path: str,
    output_dir: str,
    cat_cols: List[str],
    num_cols: List[str],
    target_col: str = 'price',
    test_size: float = 0.2,
    val_size: float = 0.125,
    random_state: int = 42,
    iqr_factor: float = 1.5
) -> None:
    """
    Main preprocessing pipeline.
    
    Args:
        input_path: Path to input CSV
        output_dir: Output directory for processed data and artifacts
        cat_cols: List of categorical column names
        num_cols: List of numerical column names
        target_col: Target column name
        test_size: Test split ratio
        val_size: Validation split ratio (from train)
        random_state: Random seed
        iqr_factor: IQR multiplier for outlier detection
    """
    # Set random seeds
    np.random.seed(random_state)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Save original price
    df['price_original'] = df[target_col].copy()
    
    # Extract date features if possible
    df = extract_date_features(df)
    
    # Create days_left if needed
    df = create_days_left(df)
    
    # Process duration
    if 'duration' in df.columns:
        df['duration_minutes'] = df['duration'].apply(parse_duration)
        if 'duration_minutes' not in num_cols:
            num_cols.append('duration_minutes')
    
    # Initial train/test split (before any fitting)
    print("Splitting train/test...")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    
    # Compute IQR thresholds ONLY on train
    print("Computing IQR thresholds on train data...")
    iqr_thresholds = compute_iqr_thresholds(df_train, num_cols, iqr_factor)
    
    # Remove outliers using train thresholds (apply to both train and test)
    print("Removing outliers...")
    before_train = len(df_train)
    df_train = remove_outliers(df_train, num_cols, iqr_thresholds)
    after_train = len(df_train)
    print(f"Train: removed {before_train - after_train} rows")
    
    before_test = len(df_test)
    df_test = remove_outliers(df_test, num_cols, iqr_thresholds)
    after_test = len(df_test)
    print(f"Test: removed {before_test - after_test} rows")
    
    # Encode categoricals (fit on train only)
    print("Encoding categoricals...")
    df_train_encoded, cat_maps = encode_categorical(df_train, cat_cols, fit=True)
    df_test_encoded, _ = encode_categorical(df_test, cat_cols, cat_maps=cat_maps, fit=False)
    
    # Scale numericals (fit on train only)
    print("Scaling numericals...")
    scaler = StandardScaler()
    df_train_encoded[num_cols] = scaler.fit_transform(df_train_encoded[num_cols])
    df_test_encoded[num_cols] = scaler.transform(df_test_encoded[num_cols])
    
    # Split train into train/val
    print("Splitting train/val...")
    df_train_final, df_val = train_test_split(
        df_train_encoded, test_size=val_size, random_state=random_state
    )
    print(f"Train: {df_train_final.shape}, Val: {df_val.shape}")
    
    # Log transform target
    df_train_final['price_log'] = np.log1p(df_train_final[target_col])
    df_val['price_log'] = np.log1p(df_val[target_col])
    df_test_encoded['price_log'] = np.log1p(df_test_encoded[target_col])
    
    # Save processed data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving processed data...")
    # Try to save as parquet, fallback to CSV if pyarrow has issues
    try:
        df_train_final.to_parquet(output_dir / 'train.parquet', index=False)
        df_val.to_parquet(output_dir / 'val.parquet', index=False)
        df_test_encoded.to_parquet(output_dir / 'test.parquet', index=False)
        print("Saved as parquet format")
    except Exception as e:
        print(f"Warning: Could not save as parquet ({e}). Saving as CSV only.")
    
    # Always save as CSV for compatibility
    df_train_final.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'val.csv', index=False)
    df_test_encoded.to_csv(output_dir / 'test.csv', index=False)
    print("Saved as CSV format")
    
    # Save artifacts
    print("Saving artifacts...")
    with open(output_dir / 'cat_maps.json', 'w') as f:
        json.dump(cat_maps, f, indent=2)
    
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'iqr_thresholds.json', 'w') as f:
        json.dump(iqr_thresholds, f, indent=2)
    
    print(f"Preprocessing complete! Outputs saved to {output_dir}")




def main():
    parser = argparse.ArgumentParser(description='Preprocess airline ticket data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--cat-cols', type=str, nargs='+', 
                       default=['airline', 'source_city', 'destination_city', 
                               'departure_time', 'arrival_time', 'stops', 'class'],
                       help='Categorical columns')
    parser.add_argument('--num-cols', type=str, nargs='+',
                       default=['duration_minutes', 'days_left'],
                       help='Numerical columns')
    parser.add_argument('--target', type=str, default='price', help='Target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--val-size', type=float, default=0.125, help='Val split ratio')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--iqr-factor', type=float, default=1.5, help='IQR multiplier')
    
    args = parser.parse_args()
    
    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        cat_cols=args.cat_cols,
        num_cols=args.num_cols,
        target_col=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        iqr_factor=args.iqr_factor
    )


if __name__ == '__main__':
    main()
