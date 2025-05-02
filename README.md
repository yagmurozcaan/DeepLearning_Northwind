# DeepLearning_Northwind

Bu proje, Northwind veritabanı üzerinde bazı tahmin modelleri geliştirmek için yapıldı. Amaç, müşterilerin satın alma davranışlarını analiz ederek gelecekteki sipariş olasılıklarını tahmin etmek.

## 📌 Proje Hakkında

Northwind veritabanındaki müşterilerin geçmiş sipariş verilerini kullanarak:

- **Sipariş Tekrarı Tahmini**: Müşterinin önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eder.
- **Yeni Ürün Satın Alma Potansiyeli**: Müşterinin yeni bir ürünü satın alma olasılığını değerlendirir.
- **İade Olasılığı Tahmini**: Müşterinin bir ürünü iade etme ihtimalini öngörür.

Bu tahminler, derin öğrenme modelleri kullanılarak gerçekleştirilmiştir.

## 🛠️ Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib
- Scikit-learn
- SMOTE (Veri dengeleme için)

## 📂 Dosya Açıklamaları

- `main.py`: Projenin ana dosyası, modellerin çalıştırılmasını sağlar.
- `order_reordering_prediction.py`: Sipariş tekrar tahmini modeli.
- `new_product_purchase_potential.py`: Yeni ürün satın alma potansiyeli modeli.
- `return_probability_prediction.py`: İade olasılığı tahmini modeli.
- `database.py`: Veritabanı bağlantı ayarları.
- `requirements.txt`: Gerekli Python kütüphaneleri.

## 🚀 Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install -r requirements.txt
2. Veritabanı bağlantı ayarlarını database.py dosyasında yapılandırın.

3. İlgili Python dosyasını çalıştırın:
    python main.py

📊 Görselleştirme
Modellerin eğitim süreci ve performansı, eğitim ve doğrulama kayıpları ile doğruluk oranları grafiklerle gösterilmektedir. Ayrıca, SMOTE uygulaması öncesi ve sonrası etiket dağılımları da görselleştirilmiştir.
