## Trading Dashboard
A Dash app for backtesting trading strategies.

### Setup
1. Clone the repo: `git clone <your-repo-url>`
2. Build Docker: `docker-compose build`
3. Run: `docker-compose up`
4. Access at `http://127.0.0.1:8050`

### Scripts
- `datalinux2.py`: Fetch Binance 1-minute data.
- `backtestrocm3.py`: Run backtests with EMA/RSI/LSTM.
- `trading_dashboard.py`: Dash app for visualization.