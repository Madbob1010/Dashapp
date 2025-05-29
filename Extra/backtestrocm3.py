#!/usr/bin/env python3

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
import glob
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure vectorbt plotting
vbt.settings.plotting['layout']['template'] = 'plotly_dark'

# Check for Linux environment (required for ROCm)
if platform.system() != "Linux":
    print(f"{Fore.RED}[ERROR] This script requires a Linux system for ROCm support. Current OS: {platform.system()}{Style.RESET_ALL}")
    sys.exit(1)

### Helper Functions

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

def train_lstm_model(close: np.ndarray, lookback: int = 30, epochs: int = 5, batch_size: int = 64):
    """Train LSTM model and return predictions with starting index."""
    try:
        X_train, y_train, X_test, _, scaler = prepare_lstm_data(close, lookback)
        start_index = lookback + len(X_train) + 1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_lstm_model(input_shape=(lookback, 1)).to(device)
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
        print(f"{Fore.CYAN}[INFO] Training LSTM model...{Style.RESET_ALL}")
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

    param_combinations = list(product(
        param_range['ema_length'],
        param_range['rsi_length'],
        param_range['rsi_threshold']
    ))

    num_processes = min(cpu_count(), len(param_combinations))
    logging.debug(f"Using {num_processes} processes for {len(param_combinations)} parameter combinations")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            compute_signals_for_params,
            [(key, df, lstm_signals) for key in param_combinations]
        )

    entries = {}
    exits = {}
    ema_dict = {}
    rsi_dict = {}

    for key, entry, exit, ema_line, rsi in results:
        entries[key] = entry
        exits[key] = exit
        ema_dict[key] = ema_line
        rsi_dict[key] = rsi

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
        logging.error(f"Portfolio creation failed: {type(e).__name__} - {e}")
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

def run_backtest(csv_file: str, plot: bool = False, use_lstm: bool = False) -> Tuple[str, Optional[Tuple], Optional[float], Dict, Optional[str]]:
    """
    Run a backtest on a single CSV and return results for Dash integration.
    
    Args:
        csv_file (str): Path to the CSV file.
        plot (bool): If True, generate and save a simplified plot.
        use_lstm (bool): If True, include LSTM signals.
    
    Returns:
        Tuple: (csv_file, best_key, best_return, portfolio, plot_filepath)
    """
    try:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        symbol = os.path.basename(csv_file).split('_')[0]
        logging.debug(f"CSV loaded for {symbol}, shape: {df.shape}")
        print(f"{Fore.CYAN}[Processing] {symbol}...{Style.RESET_ALL}")
        
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']
        if not all(col in df.columns for col in expected_columns):
            logging.warning(f"Missing expected columns in {csv_file}")
            print(f"{Fore.YELLOW}[WARN] Skipping {symbol}: Missing expected columns{Style.RESET_ALL}")
            return csv_file, None, None, {}, None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        if use_lstm:
            print(f"{Fore.CYAN}[INFO] Using LSTM signals for {symbol}{Style.RESET_ALL}")
        use_gpu = setup_pytorch() if use_lstm else False
        portfolio, best_key, best_ema, best_rsi, lstm_signals, lstm_predictions = vectorbt_backtest(df, use_lstm=use_lstm)
        
        best_return = portfolio[best_key].total_return()
        
        # Create directories for results and plots
        results_dir = "/home/madbob10/Dash/data/backtest_results/"
        plots_dir = "/home/madbob10/Dash/data/backtest_plots/"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save results
        result_file = os.path.join(results_dir, f"results_{symbol}.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump({'best_key': best_key, 'best_return': best_return, 'use_lstm': use_lstm}, f)
        print(f"{Fore.GREEN}[INFO] Results saved to {result_file}{Style.RESET_ALL}")
        
        # Generate and save simplified plot if requested
        plot_filepath = None
        if plot:
            print(f"{Fore.CYAN}[INFO] Generating simplified plot for {symbol}{Style.RESET_ALL}")
            logging.debug("Generating simplified plot")
            pf = portfolio[best_key]
            
            if pf.trades.count() == 0:
                logging.warning("No trades generated for the best parameters.")
                print(f"{Fore.YELLOW}[WARN] No trades generated for {symbol}{Style.RESET_ALL}")
            
            best_entries = (df['close'] > best_ema) & (best_rsi < best_key[2])
            best_exits = (df['close'] < best_ema) & (best_rsi > (100 - best_key[2]))
            if use_lstm and lstm_signals is not None:
                best_entries = best_entries & lstm_signals
                best_exits = best_exits & ~lstm_signals
            
            # Simplified plot: close price, portfolio value, entries/exits
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=pf.value(),
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='yellow'),
                    yaxis="y2"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index[best_entries],
                    y=df['close'][best_entries],
                    mode='markers',
                    name='Entries',
                    marker=dict(symbol='circle', color='green', size=8)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index[best_exits],
                    y=df['close'][best_exits],
                    mode='markers',
                    name='Exits',
                    marker=dict(symbol='circle', color='red', size=8)
                )
            )
            
            fig.update_layout(
                height=600,
                width=900,
                title=f"Backtest Results for {symbol} (LSTM: {use_lstm})",
                showlegend=True,
                template='plotly_dark',
                yaxis=dict(title='Price (USD)'),
                yaxis2=dict(
                    title='Portfolio Value (USD)',
                    overlaying='y',
                    side='right'
                )
            )
            
            # Save plot
            plot_filename = f"backtest_{symbol}.html"
            plot_filepath = os.path.join(plots_dir, plot_filename)
            try:
                fig.write_html(plot_filepath)
                print(f"{Fore.GREEN}[INFO] Plot saved as {plot_filepath}{Style.RESET_ALL}")
            except Exception as e:
                logging.error(f"Failed to save plot at {plot_filepath}: {e}")
                print(f"{Fore.RED}[ERROR] Could not save plot: {e}{Style.RESET_ALL}")
        
        return csv_file, best_key, best_return, portfolio, plot_filepath
    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}")
        print(f"{Fore.RED}[ERROR] Failed to process {csv_file}: {e}{Style.RESET_ALL}")
        return csv_file, None, None, {}, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest on CSV files from Binance data fetch. Requires Linux for ROCm support.")
    parser.add_argument("--plot", action="store_true", help="Generate and save simplified Plotly HTML plots")
    parser.add_argument("--use_lstm", action="store_true", help="Include LSTM model predictions in the backtest signals")
    args = parser.parse_args()
    
    data_dir = "/home/madbob10/Dash/data/"
    os.makedirs(data_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"{Fore.RED}[ERROR] No CSV files found in {data_dir}{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"{Fore.CYAN}Found {len(csv_files)} CSV files to process: {', '.join(os.path.basename(f) for f in csv_files)}{Style.RESET_ALL}")
    
    results = []
    for csv_file in csv_files:
        result = run_backtest(csv_file, plot=args.plot, use_lstm=args.use_lstm)
        results.append(result)
    
    print(f"\n{Fore.MAGENTA}=== Backtest Summary ==={Style.RESET_ALL}")
    for csv_file, best_key, best_return, portfolio, plot_filepath in results:
        symbol = os.path.basename(csv_file).split('_')[0]
        if best_key is None or best_return is None:
            print(f"{Fore.RED}[{symbol}] Failed to process{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}[{symbol}] Best Parameters: EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}, Return: {best_return*100:.2f}%{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}🎉 Backtesting complete! Check results in {data_dir}backtest_results/ and plots in {data_dir}backtest_plots/{Style.RESET_ALL}")