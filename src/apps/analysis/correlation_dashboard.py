# !/user/bin/env python
"""
Dashboard displays the cryptocurrency price correlation
"""

from apps.analysis.preprocess import load_from_data_folder_close
from apps.analysis.correlation import *
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache


__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

raw = load_from_data_folder_close()


def init_corr_dashborad(server):
    dash_app = Dash(server=server, routes_pathname_prefix='/corr/', external_stylesheets=[dbc.themes.BOOTSTRAP])
    available_cryptos = list(raw.columns)

    dash_app.layout = html.Div([
        html.H3("Cryptocurrency Close Price Correlation"),
        html.Br(),
        html.H6(f"Date Range: {raw.index.min()} to {raw.index.max()}"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="correlation")
            ],
            width=8),
            dbc.Col([
                dcc.Graph(id="matrix")
            ])
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H5("Select the cryptocurrency as x-axis"),
                dcc.Dropdown(id="x_name",
                             options=[{'label': coin, 'value': coin} for coin in available_cryptos],
                             placeholder="Select a cryptocurrency",
                             value="BTC")
            ]),
            dbc.Col([
                html.H5("Select other cryptocurrencies to display"),
                dcc.Dropdown(id="coin_display",
                             options=[{'label': coin, 'value': coin} for coin in available_cryptos],
                             value=["ETH"],
                             multi=True)
            ])
        ])
    ])
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(app):
    @app.callback(Output('correlation', 'figure'),
                  Input('x_name', 'value'),
                  Input("coin_display", "value"))
    def plot_corr(x_name, coin_display):
        scatter_data = from_merged_get_scatter_data(raw, x_name)

        coin_display_cleaned = []
        for coin in coin_display:
            if coin != x_name:
                coin_display_cleaned.append(coin)

        selected = scatter_data[scatter_data["Name"].isin(coin_display_cleaned)]

        fig = px.scatter(x=selected[x_name].values,
                         y=selected["Other"].values,
                         color=selected["Name"].values,
                         labels={
                             "x": f"{x_name} price in USD",
                             "y": f"Price in USD"
                         })

        return fig

    @app.callback(Output('matrix', 'figure'),
                  Input('x_name', 'value'),
                  Input("coin_display", "value"))
    def plot_matrix(x_name, coin_display):
        coin_display_cleaned = []
        coin_display_cleaned.extend(coin_display)
        if x_name not in coin_display_cleaned:
            coin_display_cleaned.append(x_name)

        corr = get_corr(raw[coin_display_cleaned])
        fig = px.imshow(corr.values,
                        labels={
                            "color": "Correlation"
                        },
                        x=corr.columns.values,
                        y=corr.columns.values,
                        text_auto=True)
        return fig



