import sys
import os

try:
    from src.features import generate_full_dataset
    print("Starting data generation...")
    df = generate_full_dataset('data/raw/Gold_data.csv')
    print("Dataset generated successfully!")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))
    df.to_csv('data/processed/Gold_data_featured.csv')
    print("Saved to data/processed/Gold_data_featured.csv")
except Exception as e:
    import traceback
    traceback.print_exc()
