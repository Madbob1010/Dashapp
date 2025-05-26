import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
from colorama import Fore, Style
import logging
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import os
from multiprocessing import Pool, cpu_count
from itertools import product
import pickle
import platform
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure vectorbt plotting
vbt.settings.plotting['layout']['template'] = 'plotly_dark'

# Check for Linux environment (required for ROCm)
if platform.system() != "Linux":
    print(f"{Fore.RED}[ERROR] This script requires a Linux system for ROCm support. Current OS: {platform.system()}{Style.RESET_ALL}")
    sys.exit(1)

def calculate_ema_talib(close: np.ndarray, length: int) -> np.ndarray:
    """Calculate EMA using ta-lib."""
    try:
        return talib.EMA(close, timeperiod=length)
    except Exception as e:
        logging.error(f"ta-lib EMA calculation failed: {e}")
        raise

def calculate_rsi_talib(close: np.ndarray, length: int) -> np.ndarray:
    """Calculate RSI using ta-lib."""
    try:
        return talib.RSI(close, timeperiod=length)
    except Exception as e:
        logging.error(f"ta-lib RSI calculation failed: {e}")
        raise

def setup_pytorch():
    """Initialize PyTorch with ROCm for LSTM model."""
    print(f"{Fore.MAGENTA}*** Initializing PyTorch with ROCm for AMD RX 7800XT...{Style.RESET_ALL}")
    try:
        torch_version = torch.__version__
        print(f"{Fore.CYAN}PyTorch version: {torch_version}{Style.RESET_ALL}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"{Fore.GREEN}[OK] PyTorch with ROCm setup complete: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
            logging.info(f"✔ PyTorch with ROCm initialized on {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"{Fore.YELLOW}[WARN] No GPU devices found, falling back to CPU.{Style.RESET_ALL}")
            logging.warning("No GPU devices found, using CPU.")
            return False
    except Exception as e:
        print(f"{Fore.RED}[ERROR] PyTorch with ROCm setup failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[INFO] Ensure ROCm is installed and AMD drivers are up-to-date.{Style.RESET_ALL}")
        logging.error(f"PyTorch with ROCm setup failed: {e}")
        return False

def create_lstm_model(input_shape):
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

        model = LSTMModel(input_size=1, hidden_size=32)
        return model
    except Exception as e:
        logging.error(f"LSTM model creation failed: {e}")
        raise

def prepare_lstm_data(close: np.ndarray, lookback: int = 30, train_split: float = 0.8):
    """Prepare data for LSTM training and prediction."""
    try:
        if len(close) < lookback + 1:
            raise ValueError(f"Insufficient data: {len(close)} rows, need at least {lookback + 1}")
        
        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(close_scaled) - 1):
            X.append(close_scaled[i-lookback:i])
            y.append(close_scaled[i+1])  # Predict next close price
        X, y = np.array(X), np.array(y)
        
        train_size = int(len(X) * train_split)
        if train_size < 1 or len(X) - train_size < 1:
            raise ValueError("Train/test split resulted in empty sets")
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        logging.error(f"LSTM data preparation failed: {e}")
        raise

def train_lstm_model(close: np.ndarray, lookback: int = 30, epochs: int = 5, batch_size: int = 64):
    """Train LSTM model and return predictions with starting index."""
    try:
        X_train, y_train, X_test, _, scaler = prepare_lstm_data(close, lookback)
        start_index = lookback + len(X_train) + 1  # Adjusted for next close prediction
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_lstm_model(input_shape=(lookback, 1)).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Training loop
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
        
        # Prediction
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            predictions = model(X_test).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)
        
        return predictions, start_index
    except Exception as e:
        logging.error(f"LSTM training/prediction failed: {e}")
        raise

def compute_signals_for_params(args: Tuple[Tuple[int, int, int], pd.DataFrame, Optional[pd.Series]]) -> Tuple[Tuple[int, int, int], pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute signals for a single parameter combination."""
    key, df, lstm_signals = args
    ema_length, rsi_length, rsi_threshold = key
    logging.debug(f"Processing parameters: {key}")

    try:
        ema_line = calculate_ema_talib(df['close'].values, ema_length)
        rsi = calculate_rsi_talib(df['close'].values, rsi_length)
    except Exception as e:
        logging.error(f"ta-lib calculation failed for {key}: {e}")
        raise

    if len(ema_line) != len(df) or len(rsi) != len(df):
        raise ValueError(f"Length mismatch for {key}: EMA ({len(ema_line)}), RSI ({len(rsi)}), DataFrame ({len(df)})")

    ema_line = pd.Series(ema_line, index=df.index, name=f'ema_{ema_length}').ffill()
    rsi = pd.Series(rsi, index=df.index, name=f'rsi_{rsi_length}').ffill()

    entries = (df['close'] > ema_line) & (rsi < rsi_threshold)
    exits = (df['close'] < ema_line) & (rsi > (100 - rsi_threshold))

    if lstm_signals is not None:
        entries = entries & lstm_signals
        exits = exits & ~lstm_signals

    if entries.isnull().any() or exits.isnull().any():
        raise ValueError(f"Invalid signals for {key}: Contains NaN values")

    return key, entries, exits, ema_line, rsi

def vectorbt_backtest(df: pd.DataFrame, use_lstm: bool = False, param_range: Optional[dict] = None, lookback: int = 30) -> Tuple[Dict[tuple, vbt.Portfolio], tuple, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """Run vectorbt backtest with ta-lib EMA/RSI and optional LSTM signals, using multiprocessing."""
    logging.debug("Starting vectorbt_backtest")
    if param_range is None:
        param_range = {
            'ema_length': range(15, 45, 5),
            'rsi_length': range(15, 25, 5),
            'rsi_threshold': range(30, 60, 5)
        }

    required_columns = ['close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")

    if df['close'].isnull().any():
        raise ValueError("CSV contains NaN values in 'close' column")
    if len(df) < max(param_range['ema_length']) + max(param_range['rsi_length']) + lookback:
        raise ValueError(f"CSV has insufficient data for EMA/RSI/LSTM calculations: {len(df)} rows")

    lstm_signals = None
    lstm_predictions = None

    if use_lstm:
        logging.debug("Training LSTM model")
        try:
            predictions, start_index = train_lstm_model(df['close'].values, lookback=lookback)
            lstm_predictions = pd.Series(np.nan, index=df.index, name='lstm_predictions')
            if start_index + len(predictions) > len(df):
                raise ValueError(f"Predictions (len={len(predictions)}) with start_index={start_index} exceed DataFrame length ({len(df)})")
            lstm_predictions.iloc[start_index:start_index + len(predictions)] = predictions.flatten()
            lstm_signals = (lstm_predictions > df['close']).fillna(False).astype(bool)
            logging.info(f"LSTM Buy Signals: {lstm_signals.sum()}")
            logging.info(f"LSTM Predictions sample (first 5 valid from index {start_index}): {lstm_predictions.iloc[start_index:start_index+5].to_list()}")
        except Exception as e:
            logging.error(f"LSTM signal generation failed: {e}")
            raise

    # Prepare parameter combinations
    param_combinations = list(product(
        param_range['ema_length'],
        param_range['rsi_length'],
        param_range['rsi_threshold']
    ))

    # Use multiprocessing to compute signals
    num_processes = min(cpu_count(), len(param_combinations))
    logging.debug(f"Using {num_processes} processes for {len(param_combinations)} parameter combinations")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            compute_signals_for_params,
            [(key, df, lstm_signals) for key in param_combinations]
        )

    # Collect results
    entries = {}
    exits = {}
    ema_dict = {}
    rsi_dict = {}

    for key, entry, exit, ema_line, rsi in results:
        entries[key] = entry
        exits[key] = exit
        ema_dict[key] = ema_line
        rsi_dict[key] = rsi

    # Create portfolios
    logging.debug("Creating portfolios")
    portfolio = {}
    try:
        portfolio = {
            key: vbt.Portfolio.from_signals(
                close=df['close'],
                entries=entries[key],
                exits=exits[key],
                init_cash=10000,
                freq='1min'
            )
            for key in entries.keys()
        }
    except Exception as e:
        logging.error(f"Portfolio creation failed: {e}")
        raise

    logging.debug("Selecting best portfolio")
    best_key = max(portfolio, key=lambda k: portfolio[k].total_return())
    best_entries = entries[best_key]
    best_exits = exits[best_key]
    best_ema = ema_dict[best_key]
    best_rsi = rsi_dict[best_key]

    ema_start = max(param_range['ema_length'])
    rsi_start = max(param_range['rsi_length'])
    logging.info(f"Best portfolio for {best_key}: Entries={best_entries.sum()} signals, Exits={best_exits.sum()} signals")
    logging.info(f"Best EMA sample (first 5 valid from index {ema_start}): {best_ema.iloc[ema_start:ema_start+5].to_list()}")
    logging.info(f"Best RSI sample (first 5 valid from index {rsi_start}): {best_rsi.iloc[rsi_start:rsi_start+5].to_list()}")
    logging.info(f"Best RSI stats: Min={best_rsi.min():.2f}, Max={best_rsi.max():.2f}, Mean={best_rsi.mean():.2f}")

    return portfolio, best_key, best_ema, best_rsi, lstm_signals, lstm_predictions

def process_csv(csv_file: str, plot: bool = False, use_lstm: bool = False) -> Tuple[str, Optional[Tuple], Optional[float], Optional[dict]]:
    """
    Process a CSV file for backtesting.

    Args:
        csv_file (str): Path to the CSV file (e.g., /home/user/FundedNextData/ETHUSDT.csv).
        plot (bool): If True, generate and save a Plotly HTML plot of the backtest results.
        use_lstm (bool): If True, include LSTM model predictions in the backtest signals.

    Returns:
        Tuple containing the CSV file path, best parameter key, best return, and portfolio dictionary.
    """
    logging.debug(f"Processing CSV: {csv_file}")
    try:
        logging.info(f"Starting backtest for {csv_file}")
        print(f"{Fore.CYAN}[Processing] {csv_file}...{Style.RESET_ALL}")
        
        df = pd.read_csv(csv_file)
        logging.debug(f"CSV loaded, shape: {df.shape}")
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']
        if not all(col in df.columns for col in expected_columns):
            logging.warning(f"Missing expected columns in {csv_file}")
            print(f"{Fore.YELLOW}[WARN] Skipping {csv_file}: Missing expected columns{Style.RESET_ALL}")
            return csv_file, None, None, None
        
        use_gpu = setup_pytorch() if use_lstm else False
        portfolio, best_key, best_ema, best_rsi, lstm_signals, lstm_predictions = vectorbt_backtest(df, use_lstm=use_lstm)
        
        best_return = portfolio[best_key].total_return()
        
        if plot:
            logging.debug("Generating plot")
            pf = portfolio[best_key]
            
            if pf.trades.count() == 0:
                logging.warning("No trades generated for the best parameters. Plot may lack order markers.")
                print(f"{Fore.YELLOW}[WARN] No trades generated for {csv_file}{Style.RESET_ALL}")
            
            best_entries = (df['close'] > best_ema) & (best_rsi < best_key[2])
            best_exits = (df['close'] < best_ema) & (best_rsi > (100 - best_key[2]))
            if use_lstm:
                best_entries = best_entries & lstm_signals
                best_exits = best_exits & ~lstm_signals
            
            logging.info(f"Entries: {best_entries.sum()}, Exits: {best_exits.sum()}")
            if use_lstm:
                num_lstm_signals = lstm_signals.sum() if lstm_signals is not None else 0
                logging.info(f"LSTM Buy Signals: {num_lstm_signals}")
                if num_lstm_signals == 0:
                    logging.warning("No LSTM buy signals generated.")
            
            # Create a figure with three subplots
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Price and Cumulative Results',
                    'Best EMA and RSI',
                    'LSTM Signals'
                ),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]]
            )
            
            # Panel 1: Price and Cumulative Results
            fig.add_scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='white'),
                row=1, col=1
            )
            fig.add_scatter(
                x=df.index,
                y=pf.value(),
                mode='lines',
                name='Portfolio Value',
                line=dict(color='yellow'),
                row=1, col=1,
                secondary_y=True
            )
            fig.add_scatter(
                x=df.index[best_entries],
                y=df['close'][best_entries],
                mode='markers',
                name='Entries',
                marker=dict(symbol='circle', color='green', size=6),
                row=1, col=1
            )
            fig.add_scatter(
                x=df.index[best_exits],
                y=df['close'][best_exits],
                mode='markers',
                name='Exits',
                marker=dict(symbol='circle', color='red', size=6),
                row=1, col=1
            )
            
            # Panel 2: Best EMA and RSI
            fig.add_scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='white'),
                row=2, col=1
            )
            fig.add_scatter(
                x=best_ema.index,
                y=best_ema,
                mode='lines',
                name=f'EMA {best_key[0]}',
                line=dict(color='orange'),
                row=2, col=1
            )
            fig.add_scatter(
                x=best_rsi.index,
                y=best_rsi,
                mode='lines',
                name=f'RSI {best_key[1]}',
                line=dict(color='purple'),
                row=2, col=1,
                secondary_y=True
            )
            
            # Panel 3: LSTM Signals
            if use_lstm and lstm_predictions is not None and not lstm_predictions.isna().all():
                fig.add_scatter(
                    x=lstm_predictions.index,
                    y=lstm_predictions,
                    mode='lines',
                    name='LSTM Predictions',
                    line=dict(color='magenta', dash='dash'),
                    row=3, col=1
                )
                if lstm_signals.sum() > 0:
                    fig.add_scatter(
                        x=lstm_signals.index[lstm_signals],
                        y=df['close'][lstm_signals],
                        mode='markers',
                        name='LSTM Buy Signals',
                        marker=dict(symbol='triangle-up', color='cyan', size=8),
                        row=3, col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=900,
                width=1200,
                title=f"Backtest Results for {os.path.basename(csv_file)}",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template='plotly_dark'
            )
            fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Portfolio Value (USD)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Price / EMA", row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text="RSI", row=2, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Price / Signals", row=3, col=1)
            
            # Save the plot
            plot_filename = f"portfolio_{os.path.basename(csv_file)}.html"
            plot_dir = os.path.dirname(csv_file) or '.'
            plot_path = os.path.join(plot_dir, plot_filename)
            try:
                fig.write_html(plot_path)
                print(f"{Fore.GREEN}[INFO] Plot saved as {plot_path}{Style.RESET_ALL}")
            except Exception as e:
                logging.error(f"Failed to save plot at {plot_path}: {e}")
                fallback_path = plot_filename
                try:
                    fig.write_html(fallback_path)
                    print(f"{Fore.YELLOW}[WARN] Saved plot to fallback location: {fallback_path}{Style.RESET_ALL}")
                except Exception as e2:
                    logging.error(f"Failed to save plot at fallback location {fallback_path}: {e2}")
                    print(f"{Fore.RED}[ERROR] Could not save plot: {e2}{Style.RESET_ALL}")
        
        return csv_file, best_key, best_return, portfolio
    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}")
        print(f"{Fore.RED}[ERROR] Failed to process {csv_file}: {e}{Style.RESET_ALL}")
        return csv_file, None, None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest on a CSV file with EMA/RSI and optional LSTM signals. Requires Linux for ROCm support.")
    parser.add_argument("csv_file", help="Path to CSV file (e.g., /home/user/FundedNextData/ETHUSDT.csv)")
    parser.add_argument("--plot", action="store_true", help="Generate and save a Plotly HTML plot of the backtest results (default: False)")
    parser.add_argument("--use_lstm", action="store_true", help="Include LSTM model predictions in the backtest signals (default: False)")
    args = parser.parse_args()
    
    # Example Linux-compatible path
    default_data_dir = "/home/user/FundedNextData"
    csv_file = args.csv_file if os.path.isabs(args.csv_file) else os.path.join(default_data_dir, args.csv_file)
    
    # Ensure the data directory exists
    os.makedirs(default_data_dir, exist_ok=True)
    
    process_csv(csv_file, plot=args.plot, use_lstm=args.use_lstm)