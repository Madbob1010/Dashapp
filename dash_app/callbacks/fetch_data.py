from dash import Input, Output, State
import datetime
from trading_bot.datalinux2 import update_crypto_csv
from dash_app.config.settings import DATA_DIR
import logging

def register_callback(app):
    """Register the fetch_data callback."""
    @app.callback(
        Output('fetch-status', 'children'),
        Input('fetch-button', 'n_clicks'),
        State('symbol-input', 'value'),
        State('start-date-picker', 'date'),
        State('end-date-picker', 'date')
    )
    def fetch_data(n_clicks, symbol, start_date, end_date):
        if n_clicks == 0:
            return "Enter details and click 'Fetch Data' to start."
        
        logging.debug(f"Fetching data for {symbol}, {start_date}, {end_date}")
        try:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            if start_date >= end_date:
                return "Error: Start date must be before end date."
            
            csv_path = update_crypto_csv(symbol, start_date, end_date, str(DATA_DIR))
            if csv_path:
                logging.debug(f"Data saved to {csv_path}")
                return f"Success: Data for {symbol} saved to {csv_path}"
            else:
                logging.error(f"No data fetched for {symbol}")
                return f"Error: No data fetched for {symbol}."
        except Exception as e:
            logging.error(f"Fetch error: {str(e)}")
            return f"Error: {str(e)}"