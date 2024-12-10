from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


app = FastAPI()


data = pd.read_csv("updated_customer_data_with_clusters.csv")


features = ["Income", "ShoppingFrequency", "AnnualExpenditure", "Age"]


scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features] = scaler.fit_transform(data[features])


class CustomerData(BaseModel):
    Income: float
    ShoppingFrequency: int
    AnnualExpenditure: float
    Age: int

@app.post("/predict")
def predict_cluster(customer: CustomerData):
    
    new_customer = {
        "Income": customer.Income,
        "ShoppingFrequency": customer.ShoppingFrequency,
        "AnnualExpenditure": customer.AnnualExpenditure,
        "Age": customer.Age,
    }
    new_customer_scaled = scaler.transform(pd.DataFrame([new_customer]))

    # Tính khoảng cách Euclidean
    distances = np.linalg.norm(data_scaled[features].values - new_customer_scaled, axis=1)

    # Tìm dòng gần nhất
    closest_index = distances.argmin()
    closest_customer = data.iloc[closest_index]
    cluster = closest_customer["Cluster"]

    # Trả về cụm gần nhất
    return {"cluster": int(cluster)}
