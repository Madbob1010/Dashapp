import requests
import pandas as pd
import datetime
import time
import os
import logging
from tqdm import tqdm
from colorama import init, Fore, Style


def fetch_binance_1m_data(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, limit: int = 1000) -> pd.DataFrame:
    """
    Fetching 1-minute historical kline data from Binance API for a given symbol and time range.
    """
    endpoint = "https://api.binance.com/api/v3/klines"
    data = []
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": limit,
        "startTime": int(start_date.timestamp() * 100000),
        "endTime": int(end_date.timestamp() * 100000)
    }
    # Estimating total minutes to fetch for progress bar
    total_minutes = (end_date - start_date).total_seconds() / 60
    total_requests = total_minutes / limit
    print(f"{Fore.MAGENTA}Estimating ~{total_requests:.1f} API calls for {symbol} (~{total_requests * 0.5:.1f}s){Style.RESET_ALL}")
    
    # Fetching data with progress bar
    with tqdm(total=total_minutes, desc=f"Fetching {symbol}", unit="min") as pbar:
        while params["startTime"] < params["endTime"]:
            try:
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
                if not klines:
                    break
                data.extend(klines)
                params["startTime"] = klines[-1][0] + 60000  # Next minute
                pbar.update(len(klines))
                time.sleep(0.2)  # Avoiding rate limits
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to fetch {symbol}: {e}")
                print(f"{Fore.RED}Error fetching {symbol}, retrying...{Style.RESET_ALL}")
                time.sleep(5)
                continue
    if not data:
        logging.warning(f"No data fetched for {symbol}")
        print(f"{Fore.RED}No data for {symbol}{Style.RESET_ALL}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "num_trades"])
    
    # Structuring data into DataFrame
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open", "high", "low", "close", "volume", "num_trades"]] = df[
        ["open", "high", "low", "close", "volume", "num_trades"]
    ].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume", "num_trades"]]

def validate_1m_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validating 1-minute kline data to ensure accurate candles.
    """
    df = df.rename(columns={"open_time": "timestamp"})
    time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    if not (time_diffs == 60).all():
        logging.warning("Timestamp gaps detected, forward-filling...")
        print(f"{Fore.YELLOW}Fixing timestamp gaps...{Style.RESET_ALL}")
        full_index = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="1min")
        df = df.set_index("timestamp").reindex(full_index).ffill().reset_index()
        df = df.rename(columns={"index": "timestamp"})

    duplicates = df["timestamp"].duplicated().sum()
    if duplicates > 0:
        logging.warning(f"Removing {duplicates} duplicate timestamps...")
        print(f"{Fore.YELLOW}Removing {duplicates} duplicates...{Style.RESET_ALL}")
        df = df.drop_duplicates(subset="timestamp", keep="first")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        logging.error("Negative or zero prices detected!")
        raise ValueError("Negative or zero prices detected!")
    if (df["volume"] < 0).any():
        logging.error("Negative volumes detected!")
        raise ValueError("Negative volumes detected!")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        logging.error("High price is less than max of open/close/low!")
        raise ValueError("High price is less than max of open/close/low!")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        logging.error("Low price is greater than min of open/close/high!")
        raise ValueError("Low price is greater than min of open/close/high!")

    zero_trades = df[df["num_trades"] == 0]
    if not zero_trades.empty:
        logging.info(f"Found {len(zero_trades)} candles with zero trades")
        print(f"{Fore.CYAN}{len(zero_trades)} zero-trade candles found{Style.RESET_ALL}")
    return df

def update_crypto_csv(symbol: str, csv_path: str, end_date: datetime.datetime = datetime.datetime.now()) -> None:
    """
    Updating CSV file for a cryptocurrency with new 1-minute data.
    """
    if end_date is datetime.datetime.now():
        end_date = datetime.datetime.now()

    csv_path = os.path.expanduser(csv_path)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
        logging.info(f"Existing {symbol} data: {df_existing['timestamp'].min()} to {df_existing['timestamp'].max()}")
        last_timestamp = df_existing["timestamp"].max()
        start_date = last_timestamp + datetime.timedelta(minutes=1)
    else:
        logging.info(f"No existing CSV for {symbol}, starting from 7 days ago")
        start_date = end_date - datetime.timedelta(days=7)  # Testing with 7 days
        df_existing = pd.DataFrame()

    if start_date >= end_date:
        logging.info(f"No new data to fetch for {symbol}")
        print(f"{Fore.GREEN}No new data for {symbol}{Style.RESET_ALL}")
        return

    # Fetching new data
    logging.info(f"Fetching {symbol} from {start_date} to {end_date}")
    df_new = fetch_binance_1m_data(symbol, start_date, end_date)
    if df_new.empty:
        logging.warning(f"No new data fetched for {symbol}")
        print(f"{Fore.RED}No new data for {symbol}{Style.RESET_ALL}")
        return

    # Validating and combining data
    df_new = validate_1m_data(df_new)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = validate_1m_data(df_combined)

    # Saving updated CSV
    df_combined.to_csv(csv_path, index=False)
    logging.info(f"Updated {symbol} CSV saved to {csv_path}")
    print(f"{Fore.GREEN}✔ Saved {symbol} to {csv_path}{Style.RESET_ALL}")

def fetch_multiple_cryptos(symbols: list, data_dir: str = "E:/Fund/FundedNextData/", end_date: datetime.datetime = datetime.datetime.now()) -> None:
    """
    Fetching and updating 1-minute data for multiple cryptocurrencies with progress counter.
    """
    os.makedirs(os.path.expanduser(data_dir), exist_ok=True)
    total_symbols = len(symbols)
    
    # Processing each symbol with counter
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{Fore.MAGENTA}🚀 Processing {i}/{total_symbols}: {symbol} 🚀{Style.RESET_ALL}")
        csv_path = os.path.join(os.path.expanduser(data_dir), f"{symbol}.csv")
        try:
            update_crypto_csv(symbol, csv_path, end_date)
            print(f"{Fore.GREEN}🎉 Checkpoint: {i}/{total_symbols} ({symbol}) completed!{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            print(f"{Fore.RED}❌ Error with {symbol}: {e}{Style.RESET_ALL}")
        time.sleep(1)  # Avoiding API overload

if __name__ == "__main__":
    # Defining symbols to fetch
    symbols = [
        "ETHUSDT"
    ]
    end_date = datetime.datetime.now()
    print(f"{Fore.CYAN}Starting crypto data fetch for {len(symbols)} symbols...{Style.RESET_ALL}")
    fetch_multiple_cryptos(symbols, end_date=end_date)
    print(f"{Fore.GREEN}🎉 All done! Check CSVs in E:/Fund/FundedNextData/{Style.RESET_ALL}")