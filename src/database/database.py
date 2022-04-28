# !/user/bin/env python
"""
Description of the script
"""

from flask_sqlalchemy import SQLAlchemy

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

db = SQLAlchemy()


def init_app(app):
    db.init_app(app)
    db.create_all(app)
