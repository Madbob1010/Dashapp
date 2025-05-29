from dash import Input, Output, State, html, dash
import pandas as pd
import os
from trading_bot.backtest.backtester import run_vectorbt_backtest
from dash_app.config.settings import DATA_DIR
import logging

def register_callback(app):
    """Register the run_backtest callback."""
    @app.callback(
        Output('backtest-status', 'children'),
        Output('price-chart', 'figure', allow_duplicate=True),
        Input('backtest-button', 'n_clicks'),
        State('backtest-csv-dropdown', 'value'),
        State('backtest-options', 'value'),
        prevent_initial_call=True
    )
    def run_backtest_callback(n_clicks, csv_files, options):
        logging.debug(f"Backtest triggered: files={csv_files}, options={options}")
        if n_clicks == 0:
            return "Select files and options, then click 'Run Backtest' to start.", dash.no_update
        
        if not csv_files:
            return "Error: Please select at least one CSV file or 'Entire File'.", dash.no_update
        
        try:
            use_lstm = 'use_lstm' in options
            plot_enabled = 'plot' in options
            if csv_files == ['all']:
                csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            else:
                csv_files = [os.path.join(DATA_DIR, f) for f in csv_files]
            
            results = []
            latest_fig = None
            for csv_file in csv_files:
                logging.debug(f"Running backtest on {csv_file}")
                df = pd.read_csv(csv_file)
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    logging.error(f"Invalid CSV {csv_file}: missing required columns")
                    results.append(f"{os.path.basename(csv_file).split('_')[0]}: Invalid CSV")
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isna().all():
                    logging.error(f"Invalid timestamps in {csv_file}")
                    results.append(f"{os.path.basename(csv_file).split('_')[0]}: Invalid timestamps")
                    continue
                
                result = run_vectorbt_backtest(df, use_lstm=use_lstm, plot=plot_enabled)
                logging.debug(f"Backtest result: {result}")
                portfolio, best_key, best_return, plot_fig = result
                
                symbol = os.path.basename(csv_file).split('_')[0]
                if best_key is not None:
                    result_text = (
                        f"{symbol}: Best Parameters: EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}, "
                        f"Return: {best_return*100:.2f}%"
                    )
                    results.append(result_text)
                    
                    if plot_enabled and plot_fig is not None and csv_file == csv_files[-1]:
                        latest_fig = plot_fig
                        app.latest_plot = {'type': 'backtest', 'file': csv_file, 'figure': latest_fig}
                        logging.debug("Backtest plot assigned")
                else:
                    results.append(f"{symbol}: Failed to process")
            
            logging.debug(f"Backtest results: {results}")
            return html.Ul([html.Li(r) for r in results]), latest_fig or dash.no_update
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            return f"Error: {str(e)}", dash.no_update