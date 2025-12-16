import sys
import os
import pandas as pd
from src.models import ModelTrainer
import warnings

warnings.filterwarnings('ignore')

def run():
    print("Loading data...")
    if not os.path.exists('data/processed/Gold_data_featured.csv'):
        print("Featured dataset not found. Please run test_features.py first.")
        return
        
    df = pd.read_csv('data/processed/Gold_data_featured.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    trainer = ModelTrainer(df)
    
    # Define features 
    # Must match the generated columns
    features = [
        'day_of_week', 'month', 'quarter', 'is_year_end', 
        'price_lag_1', 'price_lag_7', 'price_lag_30',
        'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
        'SPY', 'USO', 'DXY', 'USD_INR'
    ]
    
    # Keep only available features just in case
    features = [f for f in features if f in df.columns]
    
    # Train Models
    trainer.train_xgboost(features=features)
    trainer.train_prophet()
    trainer.train_lstm(features=features, epochs=10) # 10 epochs for faster testing
    
    print("\nModel Evaluation Summary:")
    print(trainer.summary_table())
    
    df_results = trainer.summary_table()
    df_results.to_csv('data/processed/model_comparison.csv')
    print("Results saved to data/processed/model_comparison.csv")
    
if __name__ == "__main__":
    run()
