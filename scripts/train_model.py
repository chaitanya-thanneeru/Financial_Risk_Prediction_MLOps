# import mlflow
# import mlflow.sklearn
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # Load dataset
# df = pd.read_csv("data/financial_data.csv")
# X = df.drop(columns=["risk_score"])
# y = df["risk_score"]

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize MLflow
# mlflow.set_tracking_uri("file:./mlflow_tracking")
# mlflow.set_experiment("Financial Risk Prediction")

# with mlflow.start_run():
#     # Train model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)

#     # Log parameters, metrics, and model
#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_metric("mse", mse)
#     # mlflow.sklearn.log_model(model, "financial_model")
#     mlflow.sklearn.log_model(model, "models/financial_risk_model")

#     print(f"✅ Model trained and logged with MSE: {mse}")


import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

# Load dataset
df = pd.read_csv("data/financial_data.csv")
X = df.drop(columns=["risk_score"])
y = df["risk_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlflow_tracking")
mlflow.set_experiment("Financial Risk Prediction")

with mlflow.start_run():
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="financial_risk_model")

    # Register new version if drift is detected
    with open("logs/drift_status.json", "r") as f:
        drift_status = json.load(f)
    
    if drift_status["drift_detected"]:
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/financial_risk_model", "financial_risk_model")
        print("✅ Drift detected! Model retrained & new version registered.")
    else:
        print("✅ No drift detected. Model remains unchanged.")
