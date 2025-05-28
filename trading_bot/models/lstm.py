import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from trading_bot.config.settings import LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE
import logging

def create_lstm_model(input_size=1, hidden_size=32, num_layers=1):
    """Create a simplified LSTM model in PyTorch."""
    try:
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=1):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(hidden_size, 16)
                self.fc2 = nn.Linear(16, 1)

            def forward(self, x):
                _, (hn, _) = self.lstm(x)
                out = self.fc1(hn[-1])
                out = self.fc2(out)
                return out

        model = LSTMModel(input_size=input_size, hidden_size=hidden_size)
        return model
    except Exception as e:
        logging.error(f"LSTM model creation failed: {e}")
        raise

def prepare_lstm_data(close: np.ndarray, lookback: int = LSTM_LOOKBACK, train_split: float = 0.8):
    """Prepare data for LSTM training and prediction."""
    try:
        if len(close) < lookback + 1:
            raise ValueError(f"Insufficient data: {len(close)} rows, need at least {lookback + 1}")
        
        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(close_scaled) - 1):
            X.append(close_scaled[i-lookback:i])
            y.append(close_scaled[i+1])
        X, y = np.array(X), np.array(y)
        
        train_size = int(len(X) * train_split)
        if train_size < 1 or len(X) - train_size < 1:
            raise ValueError("Train/test split resulted in empty sets")
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        logging.error(f"LSTM data preparation failed: {e}")
        raise

def train_lstm_model(close: np.ndarray, lookback: int = LSTM_LOOKBACK, epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE):
    """Train LSTM model and return predictions with starting index."""
    try:
        X_train, y_train, X_test, _, scaler = prepare_lstm_data(close, lookback)
        start_index = lookback + len(X_train) + 1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_lstm_model(input_size=1, hidden_size=32).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            predictions = model(X_test).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)
        
        return predictions, start_index
    except Exception as e:
        logging.error(f"LSTM training/prediction failed: {e}")
        raise