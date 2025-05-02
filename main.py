from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
# Tüm segment fonksiyonlarını import et
from order_reordering_prediction import OrderPrediction, segment_order_reordering_prediction
from return_probability_prediction import ProductReturnRiskModel, segment_return_probability_prediction
from new_product_purchase_potential import NewProductRecommendationModel, segment_new_product_purchase_potential

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hoşgeldiniz"}

@app.get("/segment/order_reordering_prediction")
def order_reordering_prediction():
    return segment_order_reordering_prediction()

@app.get("/segment/return_probability_prediction")
def return_probability_prediction():
    return segment_return_probability_prediction()

@app.get("/segment/new_product_purchase_potential")
def new_product_purchase_prediction():
    return segment_new_product_purchase_potential()

class CustomerFeatures(BaseModel):
    total_spent: float
    total_orders: int
    avg_order_value: float
    season: str  # "winter", "spring", "summer", "fall"

@app.post("/predict/order")
def predict_order(data: CustomerFeatures):
    model = OrderPrediction()
    model.load_data()
    model.create_labels()
    model.add_seasonality()
    X_train, X_test, y_train, y_test, *_ = model.prepare_data()
    model.build_model(X_train.shape[1])
    model.train_model(X_train, y_train, X_test, y_test)

    seasons = ['season_winter', 'season_spring', 'season_summer', 'season_fall']
    season_vector = [1 if f"season_{data.season.lower()}" == s else 0 for s in seasons]
    sample = np.array([[data.total_spent, data.total_orders, data.avg_order_value] + season_vector])
    prediction = model.make_prediction(sample)
    durum = "Sipariş verir" if prediction >= 0.5 else "Vermez"

    return {
        "input": data.dict(),
        "prediction": round(float(prediction), 4),
        "result": durum
    }

class ReturnRiskFeatures(BaseModel):
    discount: float
    quantity: int
    total_spent: float

@app.post("/predict/return_risk")
def predict_return_risk(data: ReturnRiskFeatures):
    model = ProductReturnRiskModel()
    model.load_data()
    model.create_labels()
    X_train, X_test, y_train, y_test = model.prepare_data()
    model.build_model(X_train.shape[1])
    class_weight_dict = {0: 1.0, 1: 4.0}
    model.train_model(X_train, y_train, class_weight_dict)

    sample = np.array([[data.discount, data.quantity, data.total_spent]])
    sample_scaled = model.scaler.transform(sample)
    prediction = model.model.predict(sample_scaled, verbose=0)[0][0]
    durum = "İade riski var" if prediction >= 0.5 else "Düşük risk"

    return {
        "input": data.dict(),
        "prediction": round(float(prediction), 4),
        "risk_level": durum
    }

class RecommendationRequest(BaseModel):
    customer_id: str
    top_n: int = 3

@app.post("/predict/recommendation")
def predict_recommendation(data: RecommendationRequest):
    model = NewProductRecommendationModel()
    model.load_data()
    model.train_model()
    result = model.recommend(data.customer_id, data.top_n)
    return {"customer_id": data.customer_id, "recommendations": result}
