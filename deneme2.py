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


class ProductReturnRiskModel:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Veritabanından gerekli veriyi çeker."""
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
        self.df = pd.read_sql_query(query, engine)

    def create_labels(self):
        """Sahte etiketler oluşturur: İade riski taşıyan ürünleri etiketler."""
        self.df['label'] = ((self.df['discount'] >= 0.2) & (self.df['total_spent'] <= 100)).astype(int)

    def prepare_data(self):
        """Özellik ve etiketleri hazırlar."""
        X = self.df[['discount', 'quantity', 'total_spent']].values
        y = self.df['label'].values

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        """Modeli oluşturur."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, class_weight_dict):
        """Modeli eğitir."""
        history = self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            class_weight=class_weight_dict,
            verbose=0
        )
        return history

    def evaluate_model(self, X_test, y_test):
        """Modelin doğruluğunu değerlendirir."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy

    def plot_metrics(self, history):
        """Eğitim ve doğrulama kayıpları ile doğruluk grafikleri çizer."""
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Eğitim Kayıp')
        plt.plot(history.history['val_loss'], label='Doğrulama Kayıp')
        plt.title('Kayıp (Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Eğitim Doğruluk')
        plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluk')
        plt.title('Doğruluk (Accuracy)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def shap_explanation(self, X_train, X_test):
        """Modelin SHAP açıklamalarını görselleştirir."""
        explainer = shap.Explainer(self.model, X_train, feature_names=['discount', 'quantity', 'total_spent'])
        shap_values = explainer(X_test[:10])

        # Tek örnek için waterfall (neden riskli gördü?)
        shap.plots.waterfall(shap_values[0], max_display=10)

        # Genel görünüm: hangi özellik ne kadar etkili
        shap.summary_plot(shap_values, features=X_test[:10], feature_names=['discount', 'quantity', 'total_spent'])

    def run(self):
        """Modeli çalıştırır ve tüm süreci başlatır."""
        self.load_data()
        self.create_labels()

        X_train, X_test, y_train, y_test = self.prepare_data()
        self.build_model(X_train.shape[1])

        # Cost-sensitive learning için sınıf ağırlığı
        class_weight_dict = {0: 1.0, 1: 4.0}  # Riskli sınıfa 4 kat daha fazla önem ver

        # Modeli eğit
        history = self.train_model(X_train, y_train, class_weight_dict)

        # Modelin başarısını değerlendir
        accuracy = self.evaluate_model(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.2f}")

        # Eğitim grafikleri
        self.plot_metrics(history)

        # SHAP açıklama
        self.shap_explanation(X_train, X_test)


# Modeli çalıştır
model = ProductReturnRiskModel()
model.run()
