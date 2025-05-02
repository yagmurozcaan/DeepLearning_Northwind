import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from database import engine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight 

class OrderPrediction:
    """Müşterilerin sipariş verip vermeyeceklerini tahmin eden sınıf."""

    def __init__(self):
        self.df = None
        self.orders_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.class_weight_dict = None

    def get_season(self, month):
        """Mevsim bilgisi döndüren fonksiyon."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def load_data(self):
        """Veritabanından gerekli veriyi çeker."""
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
        self.df = pd.read_sql_query(query, engine)
        self.df['last_order_date'] = pd.to_datetime(self.df['last_order_date'])

        # Siparişleri çek
        self.orders_df = pd.read_sql_query("SELECT customer_id, order_date FROM orders", engine)
        self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])

    def create_labels(self):
        """Etiketleri oluşturur: Müşteri 6 ay içinde tekrar sipariş verdi mi?"""
        labels = []
        for idx, row in self.df.iterrows():
            cid = row['customer_id']
            last_date = row['last_order_date']
            six_months_before = last_date - pd.DateOffset(months=6)

            customer_orders = self.orders_df[ 
                (self.orders_df['customer_id'] == cid) & 
                (self.orders_df['order_date'] >= six_months_before) & 
                (self.orders_df['order_date'] < last_date)
            ]
            label = 1 if not customer_orders.empty else 0
            labels.append(label)

        self.df['label'] = labels
        print(self.df['label'].value_counts())

    def add_seasonality(self):
        """Mevsimsellik özelliği ekler.""" 
        self.df['season'] = self.df['last_order_date'].dt.month.apply(self.get_season)
        self.df = pd.get_dummies(self.df, columns=['season'])

    def prepare_data(self):
        """Özellik ve etiketleri hazırlar.""" 
        features = ['total_spent', 'total_orders', 'avg_order_value'] + \
                   [col for col in self.df.columns if col.startswith('season_')]
        X = self.df[features].values
        y = self.df['label'].values

        # Özellikleri ölçeklendir
        X_scaled = self.scaler.fit_transform(X)

        # SMOTE ile veri dengelemesi
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_scaled, y)

        # Eğitim ve test verilerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, X, y, X_smote, y_smote

    def build_model(self, input_shape):
        """Derin öğrenme modelini oluşturur."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_test, y_test):
        """Modeli eğitir."""
        history = self.model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_data=(X_test, y_test))
        return history

    def evaluate_model(self, X_test, y_test):
        """Modelin başarısını değerlendirir."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy

    def make_prediction(self, sample):
        """Modelle tahmin yapar.""" 
        sample_scaled = self.scaler.transform(sample)
        prediction = self.model.predict(sample_scaled, verbose=0)[0][0]
        return prediction

    def visualize_smote_effect(self, y, y_smote):
        """SMOTE etkisini görselleştir.""" 
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(y, bins=2, edgecolor='black', alpha=0.7)
        plt.title("SMOTE Öncesi Sınıf Dağılımı")
        plt.xlabel('Sınıf')
        plt.ylabel('Frekans')

        plt.subplot(1, 2, 2)
        plt.hist(y_smote, bins=2, edgecolor='black', alpha=0.7)
        plt.title("SMOTE Sonrası Sınıf Dağılımı")
        plt.xlabel('Sınıf')
        plt.ylabel('Frekans')

        plt.tight_layout()
        plt.show()

    def plot_metrics(self, history):
        """Eğitim ve doğrulama kayıpları ile doğruluk grafikleri çizer."""
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_seasonality(self):
        """Mevsimsel dağılımı görselleştirir.""" 
        season_df = self.df[['season_spring', 'season_summer', 'season_winter']].copy()
        season_df['season'] = season_df.apply(
            lambda row: 'autumn' if row.sum() == 0 else
                        'spring' if row['season_spring'] == 1 else
                        'summer' if row['season_summer'] == 1 else
                        'winter', axis=1)

        plt.figure(figsize=(8, 6))
        sns.countplot(x='season', data=season_df)
        plt.title("Mevsimlere Göre Sipariş Dağılımı")
        plt.xlabel("Mevsim")
        plt.ylabel("Müşteri Sayısı")
        plt.show()

    def run(self):
        """Modeli çalıştırır ve tüm süreci başlatır."""
        self.load_data()
        self.create_labels()
        self.add_seasonality()
        
        X_train, X_test, y_train, y_test, X, y, X_smote, y_smote = self.prepare_data()
        self.build_model(X_train.shape[1])
        
        history = self.train_model(X_train, y_train, X_test, y_test)
        
        # Grafikler
        self.plot_metrics(history)
        
        # SMOTE etkisi görselleştirme
        self.visualize_smote_effect(y, y_smote)
        
        # Mevsimsel dağılımı görselleştir
        self.visualize_seasonality()
        
        # Model değerlendirme
        accuracy = self.evaluate_model(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.2f}")
        
        # Örnek tahmin
        print("\nÖrnek müşteri (2000 TL harcama, 10 sipariş, 200 TL ortalama sipariş) için mevsimsel tahminler:\n")
        seasons = ['season_winter', 'season_spring', 'season_summer', 'season_fall']
        for s in seasons:
            base_features = [2000, 10, 200]
            season_vector = [1 if col == s else 0 for col in seasons]
            sample = np.array([base_features + season_vector])
            prediction = self.make_prediction(sample)
            durum = "Sipariş verir" if prediction >= 0.5 else "Vermez"
            print(f"{s.replace('season_', '').capitalize()} → Tahmin: {prediction:.4f} → {durum}")


# Tahmin fonksiyonunu çalıştır
def segment_order_reordering_prediction():
    model = OrderPrediction()
    model.run()
    return {"message": "Sipariş tahmin modeli çalıştırıldı."}

