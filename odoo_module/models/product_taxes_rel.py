from odoo_module import db

class Product_taxes_rel(db.Model):
    __tablename__ = 'product_taxes_rel'
    prod_id = db.Column(db.Integer, primary_key= True)
    tax_id= db.Column(db.Integer, primary_key=True)

    def __init__(self,prod_id,tax_id)->None :
        self.prod_id = prod_id
        self.tax_id = tax_id