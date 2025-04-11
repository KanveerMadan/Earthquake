
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("earthquake_model.pkl")

@app.get("/")
def read_root():
    return {"message": "Earthquake Prediction API is running!"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
