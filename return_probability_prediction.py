import pandas as pd
import numpy as np
import tensorflow as tf
import shap 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sqlalchemy import create_engine
from database import engine


# Veriyi çek
query = """
SELECT 
    od.order_id,
    o.customer_id,
    od.product_id,
    od.unit_price,
    od.quantity,
    od.discount,
    od.unit_price * od.quantity * (1 - od.discount) AS total_spent
FROM 
    order_details od
JOIN orders o ON od.order_id = o.order_id
"""
df = pd.read_sql_query(query, engine)

# Sahte etiketleme: yüksek indirim ve düşük harcama → iade riski (1)
df['label'] = ((df['discount'] >= 0.2) & (df['total_spent'] <= 100)).astype(int)

# Özellikler ve hedef
X = df[['discount', 'quantity', 'total_spent']].values
y = df['label'].values

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sınıf ağırlıkları
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# TensorFlow modeli
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=0
)

# Test performansı
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Grafik: Eğitim vs Doğrulama Doğruluğu
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Doğruluk Grafiği')
plt.legend()
plt.show()

# SHAP ile açıklama
explainer = shap.Explainer(model, X_train, feature_names=['discount', 'quantity', 'total_spent'])
shap_values = explainer(X_test[:10])

# İlk örnek için SHAP waterfall grafiği
shap.plots.waterfall(shap_values[0])
