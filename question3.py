import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from database import engine

# SQL sorgusu: Müşteri bazlı kategori harcama bilgileri
query = """
    SELECT
        c.customer_id,
        cat.category_name,
        SUM(od.quantity * od.unit_price) AS total_spent
    FROM
        orders o
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, cat.category_name
"""

# Veriyi çek ve pivot tablo oluştur
df = pd.read_sql(query, engine)
pivot_df = df.pivot_table(index="customer_id", columns="category_name", values="total_spent", fill_value=0)

# Hedef etiketleri oluştur (multi-label binary)
categories = pivot_df.columns.tolist()
target_columns = [f"target_{cat}" for cat in categories]
for cat in categories:
    pivot_df[f"target_{cat}"] = (pivot_df[cat] > 0).astype(int)

# Özellikler ve hedefler
X = pivot_df[categories].values
y = pivot_df[target_columns].values

# Özellikleri ölçekle
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim/test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Çoklu çıktı verecek model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='sigmoid')  # 8 kategori için çıktı
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Test seti doğruluğu
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Doğruluğu (Kümülatif): {acc:.2f}")

# Örnek tahmin
sample_customer = X_test[0].reshape(1, -1)
prediction = model.predict(sample_customer)

# Tahmin sonuçlarını yazdır
print("\nYeni müşteri için kategori bazlı tahminler:")
for i, cat in enumerate(categories):
    print(f"{cat}: {prediction[0][i]:.2f}")
