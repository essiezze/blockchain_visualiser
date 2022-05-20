# !/user/bin/env python
"""
Description of the script
"""

from apps.analysis.predict import get_cryptos_with_available_results, PREDICTED, get_prediction_data_path, \
    get_error_data_path
import os.path
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

available_cryptos = get_cryptos_with_available_results()


def init_pred_dashborad(server):
    dash_app = Dash(server=server, routes_pathname_prefix='/prediction/', external_stylesheets=[dbc.themes.BOOTSTRAP])

    dash_app.layout = html.Div([
        html.H3("15-Day Look Ahead for Cryptocurrency Close Price"),
        html.Br(),

        dbc.Row([
            dbc.Col([
                html.H6("Select a cryptocurrency"),
                dcc.Dropdown(id="crypto",
                             options=[{'label': coin, 'value': coin} for coin in available_cryptos],
                             placeholder="Select a cryptocurrency",
                             value="BTC")
            ], width=2),
            dbc.Col([
                dcc.Graph(id="prediction"),
                html.P("For reference only")
            ])
        ])
    ])
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(app):
    @app.callback(Output('prediction', 'figure'),
                  Input('crypto', 'value'))
    def plot_corr(crypto):
        predicted = pd.read_csv(get_prediction_data_path(crypto))
        error_band = pd.read_csv(get_error_data_path(crypto))
        historical = predicted[predicted["Type"] == "Actual"]
        predicted_only = predicted[predicted["Type"] == "Predicted"]

        fig = go.Figure([
            go.Scatter(
                x=historical['Date'],
                y=historical['Close'],
                # mode="lines",
                name="Historical Price"
            ),
            go.Scatter(
                x=predicted_only["Date"],
                y=predicted_only["Close"],
                # mode="lines",
                name="Predicted Price"
            ),
            go.Scatter(
                name='Upper Bound',
                x=error_band['Date'],
                y=error_band["Upper"],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=error_band['Date'],
                y=error_band["Lower"],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])

        fig.update_layout(
            yaxis_title='Close Price (USD)',
            xaxis_title="Date",
            title='Predicted BTC price',
            hovermode="x"
        )

        fig.update_layout(height=650)

        return fig
