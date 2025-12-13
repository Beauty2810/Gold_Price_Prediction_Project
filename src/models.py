import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from prophet import Prophet
import torch
import torch.nn as nn
import pickle

def directional_accuracy(y_true, y_pred):
    """
    Calculates the directional accuracy:
    % of times the predicted direction (up/down) matches the actual direction.
    """
    y_true_dir = np.sign(np.diff(y_true))
    y_pred_dir = np.sign(np.diff(y_pred))
    
    if len(y_true_dir) == 0:
        return np.nan
        
    correct = np.sum(y_true_dir == y_pred_dir)
    return correct / len(y_true_dir)
    
def calculate_metrics(y_true, y_pred):
    return {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'DA': directional_accuracy(y_true, y_pred)
    }

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ModelTrainer:
    def __init__(self, df, target_col='price'):
        self.df = df
        self.target_col = target_col
        self.results = {}
        
    def prepare_data(self, features):
        X = self.df[features].fillna(0)
        y = self.df[self.target_col]
        return X, y
        
    def train_xgboost(self, features, n_splits=5):
        print("Training XGBoost...")
        X, y = self.prepare_data(features)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics_list = []
        model = None
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, objective='reg:squarederror')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test.values, y_pred)
            metrics_list.append(metrics)
            
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        self.results['XGBoost'] = avg_metrics
        print("XGBoost Metrics:", avg_metrics)
        return model, avg_metrics

    def train_prophet(self, n_splits=5):
        print("Training Prophet...")
        df_prophet = self.df.reset_index()
        # Ensure column for date is named ds
        if 'date' in df_prophet.columns:
            df_prophet = df_prophet.rename(columns={'date': 'ds', self.target_col: 'y'})
        else:
            df_prophet = df_prophet.rename(columns={'index': 'ds', self.target_col: 'y'})
             
        # Optional regressors
        regressors = ['SPY', 'USO', 'DXY', 'USD_INR']
        active_reg = [r for r in regressors if r in df_prophet.columns]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_list = []
        
        model = None
        for train_index, test_index in tscv.split(df_prophet):
            df_train = df_prophet.iloc[train_index]
            df_test = df_prophet.iloc[test_index]
            
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            for r in active_reg:
                model.add_regressor(r)
                
            model.fit(df_train)
            
            forecast = model.predict(df_test.drop(columns=['y']))
            y_pred = forecast['yhat'].values
            y_true = df_test['y'].values
            
            metrics = calculate_metrics(y_true, y_pred)
            metrics_list.append(metrics)
            
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        self.results['Prophet'] = avg_metrics
        print("Prophet Metrics:", avg_metrics)
        return model, avg_metrics

    def train_lstm(self, features, seq_length=10, epochs=30, n_splits=5):
        print("Training LSTM...")
        X, y = self.prepare_data(features)
        
        # Scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_length):
            X_seq.append(X_scaled[i:i+seq_length])
            y_seq.append(y_scaled[i+seq_length])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_list = []
        
        model = None
        for train_index, test_index in tscv.split(X_seq):
            X_train, X_test = X_seq[train_index], X_seq[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]
            
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            
            model = LSTMModel(input_size=X_train.shape[2], hidden_size=32, num_layers=1, output_size=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(X_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_t).numpy()
                
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_true = scaler_y.inverse_transform(y_test).flatten()
            
            metrics = calculate_metrics(y_true, y_pred)
            metrics_list.append(metrics)
            
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        self.results['LSTM'] = avg_metrics
        print("LSTM Metrics:", avg_metrics)
        return model, avg_metrics
        
    def summary_table(self):
        return pd.DataFrame(self.results).T
