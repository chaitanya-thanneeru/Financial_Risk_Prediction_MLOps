# from fastapi import FastAPI
# import mlflow.pyfunc
# import pandas as pd

# app = FastAPI()

# # Load the best MLflow model
# mlflow.set_tracking_uri("file:./mlflow_tracking")
# model_uri = "models:/financial_risk_model/1"  # Model version 1
# model = mlflow.pyfunc.load_model(model_uri)

# @app.post("/predict/")
# def predict(features: dict):
#     df = pd.DataFrame([features])
#     prediction = model.predict(df)
#     return {"risk_score": prediction.tolist()}

import mlflow
import mlflow.pyfunc
import pandas as pd
import logging
from fastapi import FastAPI

# Set MLflow Tracking URI
mlflow.set_tracking_uri("file:./mlflow_tracking")

# Load the latest registered model
model_uri = "models:/financial_risk_model/1"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()

# Set up logging
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO, format="%(asctime)s - %(message)s")

@app.post("/predict/")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]

    # Log input features and prediction
    log_entry = f"Input: {features}, Prediction: {prediction}"
    logging.info(log_entry)

    return {"risk_score": prediction}
