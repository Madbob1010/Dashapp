from dash import Dash, Input, Output, State, html, callback_context, dash
import plotly.express as px
import pandas as pd
import datetime
import os
import logging
from dash_app.config.settings import DATA_DIR
from trading_bot.datalinux2 import update_crypto_csv, reprocess_csv
from trading_bot.backtest.backtester import run_vectorbt_backtest

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def register_callbacks(app: Dash):
    """Register all Dash callbacks."""
    
    # Render tab content
    @app.callback(
        Output('tabs-content', 'children'),
        Input('tabs', 'value')
    )
    def render_content(tab):
        from dash_app.components.layout import get_data_load_content, get_data_view_content
        logging.debug(f"Rendering content for tab: {tab}")
        if tab == 'data-load':
            return get_data_load_content()
        elif tab == 'data-view':
            return get_data_view_content()
        return html.Div("Invalid tab selected")

    # Update dropdown options
    @app.callback(
        Output('symbol-dropdown', 'options'),
        Output('backtest-csv-dropdown', 'options'),
        Input('fetch-status', 'children'),
        Input('reprocess-status', 'children')
    )
    def update_dropdowns(fetch_status, reprocess_status):
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        logging.debug(f"Dropdown options: {csv_files}")
        options = [{'label': f, 'value': f} for f in csv_files]
        backtest_options = options + [{'label': 'Entire File', 'value': 'all'}]
        return options, backtest_options

    # Fetch data
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

    # Reprocess data
    @app.callback(
        Output('reprocess-status', 'children'),
        Input('reprocess-button', 'n_clicks'),
        State('reprocess-csv-dropdown', 'value'),
        State('reprocess-start-date-picker', 'date'),
        State('reprocess-end-date-picker', 'date'),
        State('reprocess-timeframe', 'value')
    )
    def reprocess_data(n_clicks, csv_file, start_date, end_date, timeframe):
        if n_clicks == 0:
            return "Select options and click 'Reprocess Data' to start."
        
        if not csv_file:
            return "Error: Please select a CSV file."
        
        logging.debug(f"Reprocessing {csv_file}, {start_date}, {end_date}, {timeframe}")
        try:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
            csv_path = os.path.join(DATA_DIR, csv_file)
            new_csv_path = reprocess_csv(csv_path, start_date, end_date, timeframe, str(DATA_DIR))
            if new_csv_path:
                logging.debug(f"Reprocessed data saved to {new_csv_path}")
                return f"Success: Reprocessed data saved to {new_csv_path}"
            else:
                logging.error("No data reprocessed")
                return "Error: No data reprocessed."
        except Exception as e:
            logging.error(f"Reprocess error: {str(e)}")
            return f"Error: {str(e)}"

    # Update price chart
    @app.callback(
        Output('price-chart', 'figure'),
        Input('symbol-dropdown', 'value'),
        Input('backtest-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def update_chart(selected_file, backtest_n_clicks):
        ctx = callback_context
        logging.debug(f"Update chart triggered: file={selected_file}, clicks={backtest_n_clicks}, triggered={ctx.triggered}")
        
        if not ctx.triggered:
            logging.debug("No trigger, returning default chart")
            return px.line(title="Select a dataset or run a backtest to view")
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        logging.debug(f"Trigger ID: {trigger_id}")
        
        if trigger_id == 'symbol-dropdown' and selected_file:
            csv_path = os.path.join(DATA_DIR, selected_file)
            logging.debug(f"Loading CSV: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                logging.debug(f"CSV loaded: rows={len(df)}, columns={df.columns.tolist()}")
                
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    logging.error(f"Missing required columns in {csv_path}: {df.columns}")
                    return px.line(title=f"Error: CSV {selected_file} missing 'timestamp' or 'close' column")
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isna().all():
                    logging.error(f"Invalid timestamps in {csv_path}")
                    return px.line(title=f"Error: Invalid timestamps in {selected_file}")
                
                if df['close'].isna().all() or not pd.api.types.is_numeric_dtype(df['close']):
                    logging.error(f"Invalid close prices in {csv_path}")
                    return px.line(title=f"Error: Invalid close prices in {selected_file}")
                
                symbol = selected_file.split('_')[0]
                logging.debug(f"Creating line plot for {symbol}")
                fig = px.line(df, x='timestamp', y='close', title=f'Close Price for {symbol}')
                fig.update_layout(template='plotly_dark')
                app.latest_plot = {'type': 'csv', 'file': csv_path, 'figure': fig}
                logging.debug("Chart created successfully")
                return fig
            except Exception as e:
                logging.error(f"Error loading {csv_path}: {str(e)}")
                return px.line(title=f"Error loading {selected_file}: {str(e)}")
        
        elif trigger_id == 'backtest-button' and app.latest_plot.get('type') == 'backtest':
            logging.debug(f"Returning backtest plot: {app.latest_plot}")
            return app.latest_plot['figure']
        
        logging.debug("Returning default chart (no valid trigger)")
        return px.line(title="Select a dataset or run a backtest to view")

    # Run backtest
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