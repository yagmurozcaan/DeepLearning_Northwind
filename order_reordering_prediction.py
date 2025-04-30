
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
        COUNT(DISTINCT o.order_id) as OrderCount,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) as TotalSpending,
        AVG(od.quantity * od.unit_price * (1 - od.discount)) as AvgOrderSize,
        o.order_date
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_details od ON o.order_id = od.order_id
        WHERE o.order_date IS NOT NULL
        GROUP BY c.customer_id, o.order_date
                
    """
    df = pd.read_sql_query(query, engine)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values(by=['customer_id', 'order_date'])
    
    df['PrevOrderDate'] = df.groupby('customer_id')['order_date'].shift(1)
    df['DaysDiff'] = (df['order_date'] - df['PrevOrderDate']).dt.days

    df['WillOrder'] = (df['DaysDiff'] <= 180).astype(int)
    df['WillOrder'] = df['WillOrder'].fillna(0) # İlk sipariş için NaN'ları 0 yap


    df_latest = df.groupby('customer_id').last().reset_index()

    df_latest['LastOrderMonth'] = df_latest['order_date'].dt.month

    print(df_latest.columns)

    #print(df)
    #Giriş özellikleri
    X = df_latest[['ordercount', 'totalspending', 'avgordersize', 'DaysDiff', 'LastOrderMonth']]
    y = df_latest['WillOrder']

    #Eksik verileri doldurma
    X = X.fillna({'totalspending': 0, 'avgordersize': 0, 'DaysDiff': 365})

    #Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #Sınıf dengesizliği için SMOTE
    smote = SMOTE(random_state=42) #Azınlık sınıf için sentetik örnekler üretir
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #Derin öğrenme modeli
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=[X_train.shape[1]]),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid') # Sınıflandırma için sigmoid
    ])

    #Model derleme
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Sınıf ağırlıkları hesaplama
    #Modelin kaybını (loss) dengesiz sınıflar için telafi eder.

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    #Model eğitimi
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        verbose=0
        )

    #Model değerlendirme
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Doğruluğu: {accuracy:.4f}")
    

    sample_customer = np.array([[1, 814.4999828338623, 20271.49999427795410, 200, 6]]) # Örnek veri: 5 sipariş, 1000 harcama, 200 avg, 90 gün, Haziran
    sample_customer_scaled = scaler.transform(sample_customer)
    prediction = model.predict(sample_customer_scaled)
    print(f"Tahmin: {prediction[0][0]:.4f} (1'e yakınsa sipariş verecek, 0'a yakınsa vermeyecek)")



order_recordering_prediction()






