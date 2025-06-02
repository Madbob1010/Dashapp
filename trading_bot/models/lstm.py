import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from trading_bot.config.settings import LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/madbob10/Dash/data/lstm.log'),
        logging.StreamHandler()
    ]
)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(data: np.ndarray, lookback: int = LSTM_LOOKBACK, epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE):
    """Train LSTM model and return predictions."""
    logging.info("Training LSTM model")
    
    # Force CPU to avoid issues
    device = torch.device("cpu")
    logging.info("Forcing CPU training for stability")
    
    try:
        # Validate input data
        if len(data) < lookback + 1:
            logging.error(f"Insufficient data for LSTM training: {len(data)} rows, need at least {lookback + 1}")
            return None, lookback
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.error("Input data contains NaN or infinite values")
            return None, lookback
        
        # Normalize data with clipping
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            logging.error("Data has zero standard deviation")
            return None, lookback
        data_normalized = np.clip((data - data_mean) / data_std, -10, 10)
        
        # Prepare data
        X, y = [], []
        for i in range(lookback, len(data_normalized)):
            X.append(data_normalized[i-lookback:i])
            y.append(data_normalized[i])
        logging.info(f"Prepared {len(X)} samples for LSTM training")
        if not X or not y:
            logging.error("No valid sequences for LSTM training")
            return None, lookback
        
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y).reshape(-1, 1)
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        logging.info(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Initialize model
        try:
            model = LSTMModel(num_layers=1, hidden_size=8, dropout=0.0).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            logging.info("Model initialized successfully")
        except Exception as e:
            logging.error(f"Model initialization failed: {e}\n{traceback.format_exc()}")
            return None, lookback
        
        # Train with error handling
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                # Reduced batch logging for brevity
                if epoch == 0:
                    logging.info(f"Batch shapes: X={batch_X.shape}, y={batch_y.shape}")
                optimizer.zero_grad()
                try:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.error(f"Invalid loss in epoch {epoch + 1}: {loss.item()}")
                        return None, lookback
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    logging.error(f"Training error in epoch {epoch + 1}: {e}\n{traceback.format_exc()}")
                    return None, lookback
            logging.info(f"Epoch {epoch + 1}/{epochs} completed, Loss: {total_loss / len(dataloader):.4f}")
        
        # Predict
        model.eval()
        with torch.no_grad():
            try:
                X_tensor = X_tensor.to(device)
                predictions = model(X_tensor).cpu().numpy()
                predictions = predictions * data_std + data_mean
                logging.info(f"Predictions shape: {predictions.shape}")
            except Exception as e:
                logging.error(f"Prediction error: {e}\n{traceback.format_exc()}")
                return None, lookback
        
        logging.info("LSTM training completed successfully")
        return predictions, lookback
    
    except Exception as e:
        logging.error(f"LSTM training failed: {e}\n{traceback.format_exc()}")
        return None, lookback