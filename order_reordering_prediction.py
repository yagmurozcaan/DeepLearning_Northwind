
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from database import engine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight


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
    SELECT
        c.customer_id,
        COUNT(o.order_id) AS total_orders,
        SUM(od.unit_price * od.quantity) AS total_spent,
        AVG(od.unit_price * od.quantity) AS avg_order_value,
        MAX(o.order_date) AS last_order_date
    FROM
        customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    GROUP BY
        c.customer_id
    """
    df = pd.read_sql_query(query, engine)
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    print(df.head())
    # ---------------------
    # 3. Tüm Sipariş Tarihlerini Çek
    # ---------------------
    orders = pd.read_sql_query("SELECT customer_id, order_date FROM orders", engine)
    orders['order_date'] = pd.to_datetime(orders['order_date'])

    # ---------------------
    # 4. Etiket (label) Oluştur — 6 Ay Öncesinde Sipariş Verdi mi?
    # ---------------------
    labels = []
    for idx, row in df.iterrows():
        cid = row['customer_id']
        last_date = row['last_order_date']
        six_months_before = last_date - pd.DateOffset(months=6)

        customer_orders = orders[
            (orders['customer_id'] == cid) &
            (orders['order_date'] >= six_months_before) &
            (orders['order_date'] < last_date)
        ]

        label = 1 if not customer_orders.empty else 0
        labels.append(label)

    df['label'] = labels
    print(df['label'].value_counts())
    # ---------------------
    # 5. Özellik ve Etiket Ayırma
    # ---------------------
    X = df[['total_spent', 'total_orders', 'avg_order_value']].values
    y = df['label'].values
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ---------------------
    # 6. TensorFlow Modeli
    # ---------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
     
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ---------------------
    # 7. Class Weight Hesaplama (Dengesiz Veri Varsa)
    # ---------------------
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # ---------------------
    # 8. Modeli Eğit
    # ---------------------
    history = model.fit(
        X_train, y_train,
        epochs=30,
        class_weight=class_weight_dict,
        verbose=0
    )

    # ---------------------
    # 9. Modeli Test Et
    # ---------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.2f}")



    sample_customer = np.array([[20, 10, 10]]) # 
    sample_customer_scaled = scaler.transform(sample_customer)
    prediction = model.predict(sample_customer_scaled)
    print(f"Tahmin: {prediction[0][0]:.4f} (1'e yakınsa sipariş verecek, 0'a yakınsa vermeyecek)")



order_recordering_prediction()






