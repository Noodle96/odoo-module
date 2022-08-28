from itertools import product
from termios import TCSADRAIN
from flask import(
    render_template, Blueprint, flash, g, redirect, request, session, url_for
)

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


from odoo_module.models.product_taxes_rel import Product_taxes_rel

from odoo_module import db
from odoo_module.models.sale_order import SaleOrder
from odoo_module.models.sale_order_line import SaleOrderLine

principal = Blueprint('principal', __name__)


@principal.route("/")
def hello_world():

    saleOrder = SaleOrder.query.all()
    saleOrderLine = SaleOrderLine.query.all()

    product = dict()
    for row in saleOrderLine:
        if not row.name in product:
            product[row.name+'.png'] = 'prediction_' + row.name + '.png'
    print(product)
    
    dataset = []


    for sale_order in saleOrder:
        for sale_order_line in saleOrderLine:
            if sale_order.id == sale_order_line.order_id:
                dataset.append(sale_order.getObject() + sale_order_line.getObject())
                #print(type(sale_order))
                break

    #print(dataset)
    #for e in dataset:
    #    print(e)



    df = pd.DataFrame (dataset, columns = ['sale_order_id', 'sale_order_name',
                                    'sale_order_dateOrder','sale_order_line_id',
                                    'sale_order_line_orderIdFK','sale_order_line_name',
                                    'sale_order_line_product_uom_qty'])

    #df.set_index("sale_order_dateOrder")
    #print(df.head())
    #print(df.info())
    plt.figure(figsize=(18,9))
    #plt.plot([1,2,3], [4,1,9],linestyle="-")
    plt.xlabel=("Dates")
    plt.ylabel=("ventas")
    #plt.show()
    #plt.savefig('odoo_module/static/test.png')


    w_001 = df[df.sale_order_line_name == "[w_001] Whisky JACK DANIEL'S Old N°7 Botella 750ml"]
    w_002 = df[df.sale_order_line_name == "[w_002] BALLANTINE X 750 ML"]
    w_003 = df[df.sale_order_line_name == "[w_003] CHIVAS REGAL 12 AÑOS 750 ML"]
    w_004 = df[df.sale_order_line_name == "[w_004] JW. BLACK 750 ML"]
    w_005 = df[df.sale_order_line_name == "[w_005] JW. DOUBLE BLACK 750 ML"]
    w_006 = df[df.sale_order_line_name == "[w_006] JW. GOLD LABEL 750 ML"]
    w_007 = df[df.sale_order_line_name == "[w_007] JW. RED 750 ML"]
    w_008 = df[df.sale_order_line_name == "[w_008] SOMETHING SPECIAL 750 ML"]
    w_009 = df[df.sale_order_line_name == "[w_009] ABSOLUT X 750 ML"]
    w_010 = df[df.sale_order_line_name == "[w_010] TANQUERAY STERLING X 750 ML"]


    plt.figure(figsize=(18,9))
    plt.plot(w_001['sale_order_dateOrder'],w_001['sale_order_line_product_uom_qty'])
    plt.savefig("odoo_module/static/[w_001] Whisky JACK DANIEL'S Old N°7 Botella 750ml.png")


    w_002["sale_order_dateOrder"] = pd.to_datetime(w_002["sale_order_dateOrder"])
    w_002 = w_002.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_002['sale_order_dateOrder'],w_002['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_002] BALLANTINE X 750 ML.png')



    w_003["sale_order_dateOrder"] = pd.to_datetime(w_003["sale_order_dateOrder"])
    w_003 = w_003.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_003['sale_order_dateOrder'],w_003['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_003] CHIVAS REGAL 12 AÑOS 750 ML.png')

    
    w_004["sale_order_dateOrder"] = pd.to_datetime(w_004["sale_order_dateOrder"])
    w_004 = w_004.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_004['sale_order_dateOrder'],w_004['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_004] JW. BLACK 750 ML.png')

    w_005["sale_order_dateOrder"] = pd.to_datetime(w_005["sale_order_dateOrder"])
    w_005 = w_005.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_005['sale_order_dateOrder'],w_005['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_005] JW. DOUBLE BLACK 750 ML.png')
    
    w_006["sale_order_dateOrder"] = pd.to_datetime(w_006["sale_order_dateOrder"])
    w_006 = w_006.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_006['sale_order_dateOrder'],w_006['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_006] JW. GOLD LABEL 750 ML.png')
    
    w_007["sale_order_dateOrder"] = pd.to_datetime(w_007["sale_order_dateOrder"])
    w_007 = w_007.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_007['sale_order_dateOrder'],w_007['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_007] JW. RED 750 ML.png')

    w_008["sale_order_dateOrder"] = pd.to_datetime(w_008["sale_order_dateOrder"])
    w_008 = w_008.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_008['sale_order_dateOrder'],w_008['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_008] SOMETHING SPECIAL 750 ML.png')

    
    w_009["sale_order_dateOrder"] = pd.to_datetime(w_009["sale_order_dateOrder"])
    w_009 = w_009.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_009['sale_order_dateOrder'],w_009['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_009] ABSOLUT X 750 ML.png')

    w_010["sale_order_dateOrder"] = pd.to_datetime(w_010["sale_order_dateOrder"])
    w_010 = w_010.sort_values(by="sale_order_dateOrder")
    plt.figure(figsize=(18,9))
    plt.plot(w_010['sale_order_dateOrder'],w_010['sale_order_line_product_uom_qty'])
    plt.savefig('odoo_module/static/[w_010] TANQUERAY STERLING X 750 ML.png')

    
    #df["sale_order_dateOrder"] = pd.to_datetime(df["sale_order_dateOrder"])
    #df = df.sort_values(by="sale_order_dateOrder")
    #plt.figure(figsize=(18,9))
    #plt.plot(df['sale_order_dateOrder'],df['sale_order_line_product_uom_qty'])
    #plt.savefig('odoo_module/static/general.png')
    #print("PRINTING DF")
    #print(df)
    



    #train_data = w_010[:len(w_010)-5]
    #print(train_data)
    #test_data = w_010[len(w_010)-5:]
    #print("test_data")
    #print(test_data)
    #test = test_data.copy()

    #convertiendo data
    #print("Info1111111111111111111111")
    #print(w_010.info())


    #                          w_001        
    w_001.sale_order_dateOrder = pd.to_datetime(w_001.sale_order_dateOrder)
    w_001.sale_order_line_product_uom_qty = pd.to_numeric(w_001.sale_order_line_product_uom_qty)
    w_001 = w_001.set_index('sale_order_dateOrder')
    
    auto_arima(w_001['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_001[:len(w_001)-5]
    test_data = w_001[len(w_001)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_001)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_001] Whisky JACK DANIEL'S Old N°7 Botella 750ml.png")


    #                          w_002       
    w_002.sale_order_dateOrder = pd.to_datetime(w_002.sale_order_dateOrder)
    w_002.sale_order_line_product_uom_qty = pd.to_numeric(w_002.sale_order_line_product_uom_qty)
    w_002 = w_002.set_index('sale_order_dateOrder')
    
    auto_arima(w_002['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_002[:len(w_002)-5]
    test_data = w_002[len(w_002)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_002)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_002] BALLANTINE X 750 ML.png")


    #                          w_003     
    w_003.sale_order_dateOrder = pd.to_datetime(w_003.sale_order_dateOrder)
    w_003.sale_order_line_product_uom_qty = pd.to_numeric(w_003.sale_order_line_product_uom_qty)
    w_003 = w_003.set_index('sale_order_dateOrder')
    
    auto_arima(w_003['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_003[:len(w_003)-5]
    test_data = w_003[len(w_003)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_003)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_003] CHIVAS REGAL 12 AÑOS 750 ML.png")


    #                          w_004   
    w_004.sale_order_dateOrder = pd.to_datetime(w_004.sale_order_dateOrder)
    w_004.sale_order_line_product_uom_qty = pd.to_numeric(w_004.sale_order_line_product_uom_qty)
    w_004 = w_004.set_index('sale_order_dateOrder')
    
    auto_arima(w_004['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_004[:len(w_004)-5]
    test_data = w_004[len(w_004)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_004)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_004] JW. BLACK 750 ML.png")


#                          w_005
    w_005.sale_order_dateOrder = pd.to_datetime(w_005.sale_order_dateOrder)
    w_005.sale_order_line_product_uom_qty = pd.to_numeric(w_005.sale_order_line_product_uom_qty)
    w_005 = w_005.set_index('sale_order_dateOrder')
    
    auto_arima(w_005['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_005[:len(w_005)-5]
    test_data = w_005[len(w_005)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_005)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_005] JW. DOUBLE BLACK 750 ML.png")

#                          w_006
    w_006.sale_order_dateOrder = pd.to_datetime(w_006.sale_order_dateOrder)
    w_006.sale_order_line_product_uom_qty = pd.to_numeric(w_006.sale_order_line_product_uom_qty)
    w_006 = w_006.set_index('sale_order_dateOrder')
    
    auto_arima(w_006['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_006[:len(w_006)-5]
    test_data = w_006[len(w_006)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_006)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_006] JW. GOLD LABEL 750 ML.png")


    #w_007
    w_007.sale_order_dateOrder = pd.to_datetime(w_007.sale_order_dateOrder)
    w_007.sale_order_line_product_uom_qty = pd.to_numeric(w_007.sale_order_line_product_uom_qty)
    w_007 = w_007.set_index('sale_order_dateOrder')
    
    auto_arima(w_007['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_007[:len(w_007)-5]
    test_data = w_007[len(w_007)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_007)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_007] JW. RED 750 ML.png")


    #w_008
    w_008.sale_order_dateOrder = pd.to_datetime(w_008.sale_order_dateOrder)
    w_008.sale_order_line_product_uom_qty = pd.to_numeric(w_008.sale_order_line_product_uom_qty)
    w_008 = w_008.set_index('sale_order_dateOrder')
    
    auto_arima(w_008['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_008[:len(w_008)-5]
    test_data = w_008[len(w_008)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_008)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_008] SOMETHING SPECIAL 750 ML.png")

    

    #w_009
    w_009.sale_order_dateOrder = pd.to_datetime(w_009.sale_order_dateOrder)
    w_009.sale_order_line_product_uom_qty = pd.to_numeric(w_009.sale_order_line_product_uom_qty)
    w_009 = w_009.set_index('sale_order_dateOrder')
    
    auto_arima(w_009['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()

    train_data = w_009[:len(w_009)-5]
    test_data = w_009[len(w_009)-5:]
    test = test_data.copy()
   
    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    arima_pred = arima_result.predict(start=len(train_data), end=len(w_009)-1,
                                        typ="levels").rename("ARIMA predictions")

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig("odoo_module/static/prediction_[w_009] ABSOLUT X 750 ML.png")




    #                          w_010
    w_010.sale_order_dateOrder = pd.to_datetime(w_010.sale_order_dateOrder)
    w_010.sale_order_line_product_uom_qty = pd.to_numeric(w_010.sale_order_line_product_uom_qty)
    w_010 = w_010.set_index('sale_order_dateOrder')
    #print("W_010 head")
    #print(w_010.head())
    #print(auto_arima(w_010['sale_order_line_product_uom_qty'],seasonal=True,
    #m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary())

    auto_arima(w_010['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary()
    print("PRINTING AUTO_ARIMA")
    print(auto_arima(w_010['sale_order_line_product_uom_qty'],seasonal=True,
    m=1,max_p=5,max_d=2,max_q=5,max_P=2,max_D=2,max_Q=2).summary())

    train_data = w_010[:len(w_010)-5]
    test_data = w_010[len(w_010)-5:]
    test = test_data.copy()
    print(train_data)
    print(test_data)
    #print("Info222222222222222")
    #print(w_010.info())

    arima_model = SARIMAX(train_data['sale_order_line_product_uom_qty'], order=(1,1,1), seasonal_order=(1,0,1,2))
    arima_result = arima_model.fit()
    arima_result.summary()
    #print("resumen arima ")
    #print(arima_result.summary())

    arima_pred = arima_result.predict(start=len(train_data), end=len(w_010)-1,
                                        typ="levels").rename("ARIMA predictions")
    #print("ARIMA PRED")
    #print(arima_pred.info())
    #print("PRINT ARIMA")
    #print(arima_pred.head())

    plt.figure(figsize=(18,9))
    test_data['sale_order_line_product_uom_qty'].plot(figsize=(16,5),legend=True)
    arima_pred.plot(legend=True)
    plt.title("Model Arima", fontsize=20)
    plt.savefig('odoo_module/static/prediction_[w_010] TANQUERAY STERLING X 750 ML.png')



    


    return render_template('index.html',
                            saleOrder=saleOrder, saleOrderLine=saleOrderLine, product=product)



@principal.route('/display/<filename>')
def displayImage(filename):
    return redirect(url_for('static', filename=filename), code=301)