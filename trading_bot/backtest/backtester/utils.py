import pickle
import logging
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from trading_bot.utils.helpers import setup_pytorch
from trading_bot.config.settings import RESULTS_DIR, PLOTS_DIR, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LOOKBACK
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceDataset(Dataset):
    """Custom Dataset for price sequences."""
    def __init__(self, close: np.ndarray, lookback: int):
        self.close = close
        self.lookback = lookback
        # Normalize data
        self.mean = np.mean(close)
        self.std = np.std(close)
        self.normalized_close = (close - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.close) - self.lookback

    def __getitem__(self, idx):
        x = self.normalized_close[idx:idx + self.lookback]
        y = self.normalized_close[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    """LSTM model for price prediction."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(close: np.ndarray, lookback: int = LSTM_LOOKBACK, epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE) -> Tuple[np.ndarray, int]:
    """Train LSTM model and return predictions."""
    logging.info(f"Training LSTM model: lookback={lookback}, epochs={epochs}, batch_size={batch_size}")
    
    # Check for sufficient data
    if len(close) < lookback + 1:
        logging.error(f"Insufficient data for LSTM: {len(close)} rows, need at least {lookback + 1}")
        raise ValueError(f"Insufficient data for LSTM: {len(close)} rows")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and setup_pytorch() else "cpu")
    logging.debug(f"Using device: {device}")

    # Prepare dataset
    dataset = PriceDataset(close, lookback)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device).unsqueeze(-1), y.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logging.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

    # Generate predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(close) - lookback):
            x = dataset.normalized_close[i:i + lookback]
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(-1)
            pred = model(x_tensor).cpu().numpy().flatten()[0]
            # Denormalize prediction
            pred = pred * dataset.std + dataset.mean
            predictions.append(pred)
    
    start_index = lookback
    logging.info(f"LSTM training complete: {len(predictions)} predictions generated")
    return np.array(predictions), start_index

def save_results(symbol: str, best_key: tuple, best_return: float, use_lstm: bool, plot_filepath: Optional[str] = None):
    """Save backtest results to a pickle file and generate an HTML report."""
    result_file = RESULTS_DIR / f"results_{symbol}.pkl"
    html_report_file = PLOTS_DIR / f"results_{symbol}.html"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save pickle file
    results_data = {
        'best_key': best_key,
        'best_return': best_return,
        'use_lstm': use_lstm,
        'plot_filepath': plot_filepath
    }
    try:
        with open(result_file, 'wb') as f:
            pickle.dump(results_data, f)
        logging.info(f"Backtest results saved to pickle: {result_file}")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Pickle file {result_file} was not created")
        if not os.access(result_file, os.R_OK):
            raise PermissionError(f"Pickle file {result_file} is not readable")
    except Exception as e:
        logging.error(f"Failed to save pickle results to {result_file}: {e}")
        raise
    
    # Generate HTML report
    try:
        html_content = f"""
        <html>
        <head><title>Backtest Results: {symbol}</title></head>
        <body style="background-color: #1E1E1E; color: #FFFFFF; font-family: Arial;">
            <h1>Backtest Results for {symbol}</h1>
            <p><b>Best Parameters:</b> EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}</p>
            <p><b>Total Return:</b> {best_return*100:.2f}%</p>
            <p><b>Used LSTM:</b> {use_lstm}</p>
            <p><b>Plot File:</b> <a href="{plot_filepath or 'None'}" style="color: #00FF00;">{plot_filepath or 'None'}</a></p>
        </body>
        </html>
        """
        with open(html_report_file, 'w') as f:
            f.write(html_content)
        logging.info(f"HTML report saved: {html_report_file}")
    except Exception as e:
        logging.error(f"Failed to save HTML report to {html_report_file}: {e}")