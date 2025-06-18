import json
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path('/home/madbob10/Dash')))

from trading_bot.config.settings import *

config = {
    "paths": {
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR)
    },
    "backtest": {
        "param_range": {
            "ema_length": list(PARAM_RANGE['ema_length']),
            "rsi_length": list(PARAM_RANGE['rsi_length']),
            "rsi_threshold": list(PARAM_RANGE['rsi_threshold'])
        }
    },
    "lstm": {
        "lookback": LSTM_LOOKBACK,
        "epochs": LSTM_EPOCHS,
        "batch_size": LSTM_BATCH_SIZE,
        "enabled": USE_LSTM
    },
    "plotting": {
        "height": PLOT_HEIGHT,
        "width": PLOT_WIDTH,
        "template": PLOT_TEMPLATE,
        "enabled": PLOT_ENABLED
    },
    "vectorbt": {
        "plotting": {
            "layout": {
                "template": VBT_SETTINGS['plotting']['layout']['template']
            }
        }
    }
}

CONFIG_DIR = Path('/home/madbob10/Dash/configs')
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_DIR / 'default.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"Created {CONFIG_DIR / 'default.json'}")