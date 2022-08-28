from time import timezone
from odoo_module import db
from sqlalchemy import DateTime

class SaleOrder(db.Model):
    __tablename__ = 'sale_order'
    id = db.Column(db.Integer, primary_key= True)
    name= db.Column(db.String)
    date_order = db.Column(db.DateTime(timezone=True))

    def __init__(self,id, name, date_order)->None :
        self.id = id
        self.name = name
        self.date_order = date_order

    def getObject(self):
        return [self.id,self.name,self.date_order]


