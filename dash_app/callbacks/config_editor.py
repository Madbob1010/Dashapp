from dash import Input, Output, State, html, dcc, dash
import json
import os
import traceback
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/madbob10/Dash/data/config.log'),
        logging.StreamHandler()
    ]
)

def flatten_config(config, prefix=''):
    flat_config = {}
    for key, value in config.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat_config.update(flatten_config(value, f"{new_key}."))
        else:
            flat_config[new_key] = value
    return flat_config

def unflatten_config(flat_config):
    config = {}
    for key, value in flat_config.items():
        parts = key.split('.')
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return config

def register_callbacks(app):
    @app.callback(
        Output('config-editor', 'children'),
        Input('config-dropdown', 'value')
    )
    def update_config_editor(config_file):
        if not config_file:
            return html.P("No config selected", style={'color': 'red'})

        config_path = Path('/home/madbob10/Dash/configs') / config_file
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config {config_file}: {e}")
            return html.P(f"Error loading config: {e}", style={'color': 'red'})

        flat_config = flatten_config(config)
        inputs = []
        for key, value in flat_config.items():
            input_id = f"config-input-{key}"
            label_style = {'color': 'white', 'margin-right': '10px'}
            if isinstance(value, bool):
                inputs.append(
                    html.Div([
                        html.Label(key, style=label_style),
                        dcc.Checklist(
                            id=input_id,
                            options=[{'label': 'Enabled', 'value': True}],
                            value=[True] if value else [],
                            style={'display': 'inline-block', 'color': 'white'}
                        )
                    ], style={'margin': '5px'})
                )
            elif isinstance(value, (list, dict)):
                inputs.append(
                    html.Div([
                        html.Label(key, style=label_style),
                        dcc.Input(
                            id=input_id,
                            type='text',
                            value=json.dumps(value),
                            style={'width': '70%', 'background-color': '#34495e', 'color': 'white', 'border': 'none', 'padding': '5px'}
                        )
                    ], style={'margin': '5px'})
                )
            else:
                inputs.append(
                    html.Div([
                        html.Label(key, style=label_style),
                        dcc.Input(
                            id=input_id,
                            type='text',
                            value=str(value),
                            style={'width': '70%', 'background-color': '#34495e', 'color': 'white', 'border': 'none', 'padding': '5px'}
                        )
                    ], style={'margin': '5px'})
                )

        return html.Div(inputs)

    @app.callback(
        Output('save-config-status', 'children'),
        Input('save-config-button', 'n_clicks'),
        State('config-save-name', 'value'),
        State('config-editor', 'children')
    )
    def save_config(n_clicks, config_name, editor_children):
        if n_clicks == 0 or not config_name:
            return html.P("Enter a config name and click Save", style={'color': 'white'})

        if not config_name.endswith('.json'):
            config_name += '.json'

        config_path = Path('/home/madbob10/Dash/configs') / config_name
        try:
            flat_config = {}
            for child in editor_children.get('props', {}).get('children', []):
                input_id = child.get('props', {}).get('children', [{}])[1].get('props', {}).get('id', '')
                value = child.get('props', {}).get('children', [{}])[1].get('props', {}).get('value', None)
                if input_id.startswith('config-input-'):
                    key = input_id.replace('config-input-', '')
                    if isinstance(value, list):
                        flat_config[key] = bool(value)
                    else:
                        try:
                            flat_config[key] = json.loads(value)
                        except json.JSONDecodeError:
                            try:
                                flat_config[key] = int(value)
                            except ValueError:
                                try:
                                    flat_config[key] = float(value)
                                except ValueError:
                                    flat_config[key] = value

            config = unflatten_config(flat_config)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logging.info(f"Saved config to {config_path}")
            return html.P(f"Config saved as {config_name}", style={'color': 'green'})
        except Exception as e:
            logging.error(f"Failed to save config {config_name}: {e}\n{traceback.format_exc()}")
            return html.P(f"Error saving config: {e}", style={'color': 'red'})