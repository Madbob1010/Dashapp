import pandas as pd
import vectorbt as vbt
import torch
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple
from trading_bot.config.settings import PARAM_RANGE, LSTM_LOOKBACK, RESULTS_DIR
from trading_bot.utils.helpers import calculate_ema_talib, calculate_rsi_talib
from trading_bot.models.lstm import train_lstm_model
import pickle
import logging
from pathlib import Path
from dash_app.utils.plotting import generate_plot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify').setLevel(logging.WARNING)

def compute_signals_for_params(args: Tuple[Tuple[int, int, int], pd.DataFrame, Optional[pd.Series]]) -> Tuple[Tuple[int, int, int], pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute signals for a single parameter combination."""
    key, df, lstm_signals = args
    ema_length, rsi_length, rsi_threshold = key
    logging.info(f"Processing parameters: {key}")

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

# In trading_bot/backtest/backtester/backtest.py
def run_vectorbt_backtest(
    df: pd.DataFrame,
    symbol: str,
    use_lstm: bool = False,
    plot: bool = False,
    param_range: Optional[dict] = None,
    lookback: int = LSTM_LOOKBACK
) -> Tuple[Optional[vbt.Portfolio], Optional[tuple], float, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    """Run vectorized backtest with ta-lib EMA/RSI and optional LSTM signals."""
    logging.info(f"Starting vectorbt_backtest for {symbol}")
    if param_range is None:
        param_range = PARAM_RANGE

    required_columns = ['close']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"DataFrame missing required columns: {required_columns}")
        return None, None, -float('inf'), None, None, None, None

    min_rows = max(param_range['ema_length']) + max(param_range['rsi_length']) + (lookback if use_lstm else 0)
    if len(df) < min_rows:
        logging.error(f"Insufficient data: {len(df)} rows, need at least {min_rows} for EMA/RSI{'/LSTM' if use_lstm else ''}")
        return None, None, -float('inf'), None, None, None, None

    if len(df) < max(param_range['ema_length']) + max(param_range['rsi_length']) + lookback:
        logging.error(f"Insufficient data for EMA/RSI/LSTM: {len(df)} rows")
        return None, None, -float('inf'), None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.randn(10, 10, device=device)
        logging.info(f"Using device: {device}")
    except RuntimeError:
        device = torch.device("cpu")
        logging.info("Falling back to CPU")

    lstm_signals = None
    lstm_predictions = None
    if use_lstm:
        logging.info("Training LSTM model")
        try:
            predictions, start_index = train_lstm_model(df['close'].values, lookback=lookback)
            lstm_predictions = pd.Series(np.nan, index=df.index, name='lstm_predictions')
            if start_index + len(predictions) > len(df):
                logging.error(f"Predictions (len={len(predictions)}) with start_index={start_index} exceed DataFrame length ({len(df)})")
                return None, None, -float('inf'), None, None, None, None
            lstm_predictions.iloc[start_index:start_index + len(predictions)] = predictions.flatten().tolist()
            lstm_signals = (lstm_predictions > df['close']).astype(bool).fillna(False)
            logging.info(f"LSTM Buy Signals: {lstm_signals.sum()}")
            logging.info(f"LSTM Predictions sample: {lstm_predictions.iloc[start_index:start_index+2].to_list()}")
        except Exception as e:
            logging.error(f"LSTM signal generation failed: {e}")
            lstm_predictions = None
            lstm_signals = None

    param_combinations = list(product(
        param_range['ema_length'],
        param_range['rsi_length'],
        param_range['rsi_threshold']
    ))

    num_processes = min(cpu_count(), len(param_combinations))
    logging.info(f"Using {num_processes} processes for {len(param_combinations)} parameter combinations")
    
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

    logging.info("Creating portfolios")
    portfolio = {}
    try:
        portfolio = {
            key: vbt.Portfolio.from_signals(
                close=df['close'],
                entries=entries[key],
                exits=exits[key],
                init_cash=10000,
                freq='1m' if '1m' in symbol else '15min'
            )
            for key in entries
        }
    except Exception as e:
        logging.error(f"Portfolio creation failed: {type(e).__name__} - {e}")
        return None, None, -float('inf'), None, None, None, None

    logging.info("Selecting best portfolio")
    best_key = None
    best_portfolio = None
    best_return = -float('inf')
    best_entries = None
    best_exits = None
    try:
        best_key = max(portfolio, key=lambda k: portfolio[k].total_return())
        best_portfolio = portfolio[best_key]
        best_entries = entries[best_key]
        best_exits = exits[best_key]
        best_return = best_portfolio.total_return()
    except Exception as e:
        logging.error(f"Best portfolio selection failed: {e}")
        return None, None, -float('inf'), None, None, None, None

    ema_start = max(param_range['ema_length'])
    rsi_start = max(param_range['rsi_length'])
    logging.info(f"Best portfolio for {best_key}: Entries={best_entries.sum()} signals, Exits={best_exits.sum()} signals")
    logging.info(f"Best EMA sample (first 5 valid from index {ema_start}): {ema_dict[best_key].iloc[ema_start:ema_start+5].to_list()}")
    logging.info(f"Best RSI sample (first 5 valid from index {rsi_start}): {rsi_dict[best_key].iloc[rsi_start:rsi_start+5].to_list()}")
    logging.info(f"Best RSI stats: Min={rsi_dict[best_key].min():.2f}, Max={rsi_dict[best_key].max():.2f}, Mean={rsi_dict[best_key].mean():.2f}")

    plot_filepath = None
    if plot:
        logging.info("Generating backtest plot")
        try:
            plot_filepath = generate_plot(
                symbol=symbol,
                df=df,
                portfolio=best_portfolio,
                best_entries=best_entries,
                best_exits=best_exits,
                use_lstm=use_lstm,
                lstm_predictions=lstm_predictions
            )
        except Exception as e:
            logging.error(f"Plot generation failed for {symbol}: {e}")
            plot_filepath = None

    return best_portfolio, best_key, best_return, best_entries, best_exits, lstm_predictions, plot_filepath

def save_results(symbol: str, best_key: tuple, best_return: float, use_lstm: bool, plot_filepath: Optional[str]):
    """Save backtest results to a pickle file."""
    result_file = RESULTS_DIR / f"results_{symbol}.pkl"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(result_file, 'wb') as f:
        pickle.dump({'best_key': best_key, 'best_return': best_return, 'use_lstm': use_lstm, 'plot_filepath': plot_filepath}, f)
    logging.info(f"Backtest results saved to pickle: {result_file}")