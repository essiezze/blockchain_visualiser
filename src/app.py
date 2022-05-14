# !/user/bin/env python
"""
The launching script of the flask application
"""

from flask import Flask
from apps.analysis.price_dashboard import init_dashborad

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

CONFIG_PATH = "./config.py"


def create_app():
    app = Flask(__name__)
    # app.config.from_object(CONFIG_PATH)
    return app


if __name__ == "__main__":
    app = create_app()
    price_dashboard = init_dashborad(app)
    app.run("0.0.0.0")

