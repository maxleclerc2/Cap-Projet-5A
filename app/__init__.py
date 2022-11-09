from flask import Flask
from flask_caching import Cache
from flask_bootstrap import Bootstrap
from flask_cors import CORS
import logging


# -----------------------------------------------------------------------------
# Flask init
# -----------------------------------------------------------------------------

def init_app():
    logging.basicConfig(filename='ssa.log',
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        level=logging.INFO)
    logging.info('Started SSA Web App Prototype')

    app = Flask(__name__, instance_relative_config=False)
    CORS(app)
    app.config.from_pyfile("config.py")

    bootstrap = Bootstrap(app)
    cache = Cache(app)

    with app.app_context():
        from . import routes, api, functions

        return app
