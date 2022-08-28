from odoo_module import db

class SaleOrderLine(db.Model):
    __tablename__ = "sale_order_line"
    id = db.Column(db.Integer, primary_key= True)
    order_id = db.Column(db.Integer)
    name = db.Column(db.String)
    product_uom_qty = db.Column(db.Integer)

    def __init__(self,id,order_id, name,product_uom_qty) -> None:
        self.id = id
        self.order_id = order_id
        self.name = name
        self.product_uom_qty = product_uom_qty

    def getObject(self):
        return [self.id,self.order_id,self.name,self.product_uom_qty]