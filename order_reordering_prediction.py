import pandas as pd
from datetime import timedelta
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from database import engine

"""görev:Northwind veritabanında 
müşterilerin toplam harcaması,
sipariş sayısı ve
ortalama sipariş büyüklüğüne 
göre 
bir müşterinin önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini 
tahmin eden bir derin öğrenme modeli kur.

İpucu: Veritabanından Orders, Order Details, Customers tablolarını kullan.

"Son sipariş tarihi" bilgisine göre 6 ay sınırı belirle.
"""

def order_recordering_prediction():
    query = """
    select
        c.customer_id,
        max(o.order_date) as LastOrderDate,
        count(distinct o.order_id) as OrderCount,
        sum(od.unit_price * od.quantity * (1 - od.discount)) as TotalSpent,
        avg(od.unit_price * od.quantity * (1 - od.discount)) as AvgOrderSize
    
    from customers c
    join orders o on c.customer_id = o.customer_id
    join order_details od on o.order_id = od.order_id

    group by c.customer_id,o.order_date
        
    """
    df = pd.read_sql_query(query, engine)
    print(df)

order_recordering_prediction()