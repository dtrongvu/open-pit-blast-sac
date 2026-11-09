"""
Step 1: True Spatio-Temporal Feature Engineering (Zero Leakage + Calibration)
=============================================================================
This module loads raw open-pit blast fragmentation data, performs chronological 
data partitioning (70% Train, 15% Validation, 15% Test), constructs a 
bench-constrained temporal memory window (90-day sliding horizon), and applies 
a sequential cold-start fallback strategy to eliminate data leakage.

Reference Paper: "Optimizing Open-Pit Blast Design via Soft Actor-Critic Reinforcement Learning"
Target Journal: Mining, Metallurgy & Exploration (Springer Nature)
"""

import os
import io
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Robust loading and initial preprocessing of historical blast fragmentation CSV data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")
        
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    for i, line in enumerate(lines):
        if b'DATE' in line and i < 5:
            idx = line.find(b'DATE')
            cleaned_lines.append(line[idx:].decode('utf-8', errors='ignore'))
        elif len(cleaned_lines) > 0:
            cleaned_lines.append(line.decode('utf-8', errors='ignore'))
            
    df = pd.read_csv(io.StringIO("".join(cleaned_lines)))
    df.columns = [col.replace('\r', '').strip() for col in df.columns]
    
    # Standardize column naming conventions
    if 'Bench elevation' in df.columns:
        df.rename(columns={'Bench elevation': 'BENCH_ELEVATION'}, inplace=True)
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'DATE'}, inplace=True)
        
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Assign geological domain based on bench elevation thresholds
    def assign_zone(elevation):
        if elevation >= 30:
            return 'Upper_Zone'
        elif elevation >= -30:
            return 'Mid_Zone'
        else:
            return 'Deep_Zone'
            
    df['DOMAIN'] = df['BENCH_ELEVATION'].apply(assign_zone)
    return df

def calculate_temporal_memory(df_target: pd.DataFrame, df_history_pool: pd.DataFrame, t_window: pd.Timedelta) -> pd.DataFrame:
    """
    Calculates rolling geomechanical expectations (mu_A, mu_C_n) and volatilities 
    (sigma_A, sigma_C_n) strictly within a w-day bench-constrained temporal memory window.
    """
    df_target = df_target.copy()
    mu_A, sigma_A, mu_C_n, sigma_C_n = [], [], [], []
    
    for i in range(len(df_target)):
        current_date = df_target.loc[i, 'DATE']
        current_bench = df_target.loc[i, 'BENCH_ELEVATION']
        
        # Isolate strictly by time window and identical bench elevation (Zero Data Leakage)
        history = df_history_pool[
            (df_history_pool['DATE'] < current_date) &
            (df_history_pool['DATE'] >= (current_date - t_window)) &
            (df_history_pool['BENCH_ELEVATION'] == current_bench)
        ].copy()
        
        if len(history) == 0:
            mu_A.append(np.nan)
            sigma_A.append(np.nan)
            mu_C_n.append(np.nan)
            sigma_C_n.append(np.nan)
        else:
            mu_A.append(history['ROCK_FACTOR_A'].mean())
            sigma_A.append(history['ROCK_FACTOR_A'].std() if len(history) > 1 else 0.0)
            mu_C_n.append(history['C_n'].mean())
            sigma_C_n.append(history['C_n'].std() if len(history) > 1 else 0.0)
            
    df_target['mu_A'] = mu_A
    df_target['sigma_A'] = sigma_A
    df_target['mu_C_n'] = mu_C_n
    df_target['sigma_C_n'] = sigma_C_n
    
    return df_target

def execute_feature_engineering(file_path: str, window_days: int = 90):
    """
    Executes the full spatio-temporal feature engineering pipeline:
    1. Chronological Sorting and 70/15/15 Split
    2. Rolling Bench-Constrained Temporal Memory Calculation (90 Days)
    3. Sequential Cold-Start Fallback Imputation
    """
    print(f"Loading data from {file_path}...")
    df_clean = load_and_preprocess_data(file_path)
    
    # 1. Chronological Sorting & Partitioning (70% Train, 15% Val, 15% Test)
    df_clean = df_clean.sort_values(by='DATE').reset_index(drop=True)
    train_idx = int(len(df_clean) * 0.70)
    val_idx = int(len(df_clean) * 0.85)
    
    train_df_raw = df_clean.iloc[:train_idx].copy().reset_index(drop=True)
    val_df_raw = df_clean.iloc[train_idx:val_idx].copy().reset_index(drop=True)
    test_df_raw = df_clean.iloc[val_idx:].copy().reset_index(drop=True)
    
    temporal_window = pd.Timedelta(days=window_days)
    
    # 2. Rolling Memory Window Calculation
    print("Calculating Temporal Memory for Training Set...")
    train_df = calculate_temporal_memory(train_df_raw, train_df_raw, temporal_window)
    
    print("Calculating Temporal Memory for Validation Set...")
    train_val_pool = pd.concat([train_df_raw, val_df_raw]).reset_index(drop=True)
    val_df = calculate_temporal_memory(val_df_raw, train_val_pool, temporal_window)
    
    print("Calculating Temporal Memory for Test Set...")
    full_pool = pd.concat([train_df_raw, val_df_raw, test_df_raw]).reset_index(drop=True)
    test_df = calculate_temporal_memory(test_df_raw, full_pool, temporal_window)
    
    # 3. Sequential Cold-Start Imputation
    # Fallback 1: Train & Validation (Knowledge derived strictly from Training Split)
    train_mu = train_df['ROCK_FACTOR_A'].mean()
    train_sigma = train_df['ROCK_FACTOR_A'].std()
    train_mu_Cn = train_df['C_n'].mean()
    train_sigma_Cn = train_df['C_n'].std()
    
    train_df['mu_A'].fillna(train_mu, inplace=True)
    train_df['sigma_A'].fillna(train_sigma, inplace=True)
    train_df['mu_C_n'].fillna(train_mu_Cn, inplace=True)
    train_df['sigma_C_n'].fillna(train_sigma_Cn, inplace=True)
    
    val_df['mu_A'].fillna(train_mu, inplace=True)
    val_df['sigma_A'].fillna(train_sigma, inplace=True)
    val_df['mu_C_n'].fillna(train_mu_Cn, inplace=True)
    val_df['sigma_C_n'].fillna(train_sigma_Cn, inplace=True)
    
    # Fallback 2: Test Set (Knowledge cumulative up to end of Validation Split)
    train_val_history = pd.concat([train_df, val_df])
    test_mu = train_val_history['ROCK_FACTOR_A'].mean()
    test_sigma = train_val_history['ROCK_FACTOR_A'].std()
    test_mu_Cn = train_val_history['C_n'].mean()
    test_sigma_Cn = train_val_history['C_n'].std()
    
    test_df['mu_A'].fillna(test_mu, inplace=True)
    test_df['sigma_A'].fillna(test_sigma, inplace=True)
    test_df['mu_C_n'].fillna(test_mu_Cn, inplace=True)
    test_df['sigma_C_n'].fillna(test_sigma_Cn, inplace=True)
    
    print(f"\n--- Feature Engineering Summary ---")
    print(f"Total Cleaned Blasts: {len(df_clean)}")
    print(f"Training Environment (Episodes): {len(train_df)}")
    print(f"Validation Environment (Episodes): {len(val_df)}")
    print(f"Test Environment / Shadow Replay (Episodes): {len(test_df)}")
    print("Spatio-temporal state space features generated successfully with zero data leakage!")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Spatio-Temporal Feature Engineering")
    parser.add_argument("--file_path", type=str, default="data/27_7_fragmentation.csv", help="Path to raw CSV dataset")
    parser.add_argument("--window_days", type=int, default=90, help="Sliding temporal memory window duration in days")
    args = parser.parse_args()
    
    if os.path.exists(args.file_path):
        train_df, val_df, test_df = execute_feature_engineering(args.file_path, args.window_days)
    else:
        print(f"[INFO] File '{args.file_path}' not found locally. Step 1 module initialized successfully for pipeline import.")
