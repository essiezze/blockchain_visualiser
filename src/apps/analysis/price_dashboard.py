# !/user/bin/env python
"""
Description of the script
"""

from apps.analysis.preprocess import load_from_data_folder_all
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import plotly.express as px

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

PRICE_TYPES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FULL_NAMES = {
    "Open": "Open Price",
    "High": "High Price",
    "Low": "Low Price",
    "Close": "Close Price",
    "Adj Close": "Adjusted Close Price",
    "Volume": "Volume"
}

all = load_from_data_folder_all()


def init_dashborad(server):
    dash_app = Dash(server=server, routes_pathname_prefix='/price/', external_stylesheets=[dbc.themes.BOOTSTRAP])

    available_coins = all["Coin"].unique()

    dash_app.layout = html.Div([
        html.H3("Cryptocurrency Price"),
        dcc.Graph(id="price"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H6("Select a cryptocurrency"),
                dcc.Dropdown(id="coin_dropdown",
                             options=[{'label': coin, 'value': coin} for coin in available_coins],
                             placeholder="Select a cryptocurrency",
                             value="BTC"),
                html.Br()
            ]),
            dbc.Col([
                html.H6("Select a price type"),
                dcc.Dropdown(id="price_type_dropdown",
                             options=[{'label': full_name, 'value': p_type}
                                      for p_type, full_name in FULL_NAMES.items()],
                             placeholder="Select a price type",
                             value="Close"),
                html.Br()
            ])
        ])
    ])
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(app):
    @app.callback(Output('price', 'figure'),
                  Input('coin_dropdown', 'value'),
                  Input('price_type_dropdown', 'value'))
    def plot_price(coin, price_type):
        if not coin:
            coin = "BTC"

        selected = all[all["Coin"] == coin]

        fig = px.line(selected, x="Date", y=price_type, title=f"The {FULL_NAMES[price_type]} of {coin}")
        fig.update_xaxes(rangeslider_visible=True)
        return fig


