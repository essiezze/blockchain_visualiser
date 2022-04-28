# !/user/bin/env python
"""
The launching script of the flask application
"""

from flask import Flask

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

CONFIG_PATH = "./config.py"


def create_app():
    app = Flask(__name__)
    app.config.from_object(CONFIG_PATH)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run("0.0.0.0")

