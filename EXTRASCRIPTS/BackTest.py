
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

# Existing imports and functions (calculate_ema_talib, calculate_rsi_talib, setup_tensorflow,
# create_lstm_model, prepare_lstm_data, train_lstm_model) remain unchanged.
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure vectorbt plotting
vbt.settings.plotting['layout']['template'] = 'plotly_dark'

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


    # Collect results
    entries = {}
    exits = {}
    ema_dict = {}
    rsi_dict = {}

    # Generate results by iterating over all parameter combinations
    results = []
    for ema_length in param_range['ema_length']:
        ema_line = pd.Series(calculate_ema_talib(df['close'].values, ema_length), index=df.index)
        for rsi_length in param_range['rsi_length']:
            rsi = pd.Series(calculate_rsi_talib(df['close'].values, rsi_length), index=df.index)
            for rsi_threshold in param_range['rsi_threshold']:
                entry = (df['close'] > ema_line) & (rsi < rsi_threshold)
                exit = (df['close'] < ema_line) & (rsi > (100 - rsi_threshold))
                key = (ema_length, rsi_length, rsi_threshold)
                results.append((key, entry, exit, ema_line, rsi))

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

    return portfolio, best_key, best_ema, best_rsi, 

def process_csv(csv_file: str, plot: bool = False, use_lstm: bool = False) -> Tuple[str, Optional[Tuple], Optional[float], Optional[dict]]:
    """Process a CSV file for backtesting."""
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
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest on CSV file")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("--plot", action="store_true", help="Plot the best portfolio")
    parser.add_argument("--use_lstm", action="store_true", help="Use LSTM model for signals")
    args = parser.parse_args()
    
    process_csv(args.csv_file, plot=args.plot, use_lstm=args.use_lstm)