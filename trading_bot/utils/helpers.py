import logging
import platform
import sys
import torch
from colorama import Fore, Style
import talib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_system():
    """Check if the system is Linux for ROCm support."""
    if platform.system() != "Linux":
        print(f"{Fore.RED}[ERROR] This script requires a Linux system for ROCm support. Current OS: {platform.system()}{Style.RESET_ALL}")
        sys.exit(1)

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