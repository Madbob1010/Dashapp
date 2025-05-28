import pandas as pd
import os
from config.settings import DATA_DIR
from colorama import Fore, Style
import logging

def load_and_validate_data(csv_file: str) -> tuple[pd.DataFrame, str]:
    """Load and validate CSV data."""
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
            return None, symbol
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        if df['close'].isnull().any():
            raise ValueError("CSV contains NaN values in 'close' column")
        
        return df, symbol
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {e}")
        print(f"{Fore.RED}[ERROR] Failed to process {csv_file}: {e}{Style.RESET_ALL}")
        return None, None