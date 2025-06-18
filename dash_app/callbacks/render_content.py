from dash import Input, Output, html
import logging

def register_callback(app):
    """Register the render_content callback."""
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