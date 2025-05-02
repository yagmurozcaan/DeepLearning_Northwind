# DeepLearning_Northwind

This project was developed to build some prediction models using the Northwind database. The goal is to analyze customer purchasing behavior and predict the likelihood of future orders.

## 📌 About the Project

Using customers' past order data from the Northwind database:

- **Order Reordering Prediction**: Predicts whether a customer will reorder in the next 6 months.
- **New Product Purchase Potential**: Evaluates the likelihood of a customer purchasing a new product.
- **Return Probability Prediction**: Estimates the chance of a product being returned.

These predictions are made using deep learning models.

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Pandas, NumPy, Matplotlib  
- Scikit-learn  
- SMOTE (for data balancing)

## 📂 File Descriptions

- `main.py`: Main file that runs the models.
- `order_reordering_prediction.py`: Model for predicting reorders.
- `new_product_purchase_potential.py`: Model for predicting new product purchases.
- `return_probability_prediction.py`: Model for predicting return probability.
- `database.py`: Database connection settings.
- `requirements.txt`: Python dependencies.

## 🚀 How to Run

1. Install required libraries:

   ```bash
   pip install -r requirements.txt
2.Set your database connection settings in database.py.

3. Run the desired Python file:
      ```bash
   python main.py
📊 Visualization
The training and performance of the models are shown with accuracy and loss graphs for both training and validation. SMOTE class balancing effects before and after are also visualized.

   
