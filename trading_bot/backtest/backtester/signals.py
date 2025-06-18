import pandas as pd
import logging
from typing import Tuple, Optional
from trading_bot.utils.helpers import calculate_ema_talib, calculate_rsi_talib

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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