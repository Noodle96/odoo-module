from flask import(
    render_template, Blueprint, flash, g, redirect, request, session, url_for
)
from odoo_module.models.product_taxes_rel import Product_taxes_rel

from odoo_module import db

principal = Blueprint('principal', __name__)


@principal.route("/")
def hello_world():
    product_taxes_rel = Product_taxes_rel.query.all()
    return render_template('index.html',product_taxes_rel=product_taxes_rel)

    