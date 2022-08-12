# configuracion de produccion modo desarrollo
from asyncio.subprocess import DEVNULL
from flask_sqlalchemy import SQLAlchemy

class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True

    # Configuracion de la base de datos
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    #SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:Angel1612@localhost:3306/huella_carbono"
    SQLALCHEMY_DATABASE_URI = "postgresql://odoo:odoo@localhost:5432/database_odoo"


class ProductionConfig(Config):
    DEBUG = False

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    SECRET_KEY = 'dev'
    DEBUG = True

class TestingConfig(Config):
    TESTING = True