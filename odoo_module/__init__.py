from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')
app.jinja_env.filters['zip'] = zip

db = SQLAlchemy(app)

from odoo_module.view.principal import principal
app.register_blueprint(principal)


db.create_all()