# debug_popsign.py  — run this BEFORE anything else
import pandas as pd
import numpy as np
import glob, os

KAGGLE_DIR =r"C:\Users\524ta\Downloads\asl-signs"   # ← same path as your extraction script

# Find one parquet file
parquet_files = glob.glob(f"{KAGGLE_DIR}/**/*.parquet", recursive=True)
print(f"Found {len(parquet_files)} parquet files")
print(f"Example path: {parquet_files[0]}")

# Load one and inspect it
df = pd.read_parquet(parquet_files[0])
print(f"\nShape: {df.shape}")
print(f"\nAll columns:\n{list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nNull counts:\n{df.isnull().sum().sum()} total nulls")
print(f"\nSample values (first row):\n{df.iloc[0]}")