import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
import colorama
import sklearn
import requests
import tqdm
import dash

print("All packages imported successfully!")
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"VectorBT Version: {vbt.__version__}")
print(f"TA-Lib Version: {talib.__version__}")
print(f"Colorama Version: {colorama.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"Requests Version: {requests.__version__}")
print(f"TQDM Version: {tqdm.__version__}")
print(f"Dash Version: {dash.__version__}")