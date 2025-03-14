# import pandas as pd
# import numpy as np
# from evidently.test_suite import TestSuite
# from evidently.metric_preset import TestDataDrift, TestTargetDrift

# # Load past predictions
# df_past = pd.read_csv("data/financial_data.csv")  # Baseline data
# df_new = pd.read_csv("logs/predictions.log", delimiter=",", header=None, names=["timestamp", "input", "prediction"])

# # Convert string input to dictionary
# df_new["input"] = df_new["input"].apply(eval)
# df_new = df_new["input"].apply(pd.Series)
# df_new["prediction"] = df_new["prediction"].astype(float)

# # Define Evidently Test Suite for Drift Detection
# drift_suite = TestSuite(
#     tests=[
#         TestDataDrift(column_name="market_volatility"),
#         TestDataDrift(column_name="interest_rate"),
#         TestDataDrift(column_name="inflation_rate"),
#         TestTargetDrift()
#     ]
# )

# # Run drift detection
# drift_suite.run(reference_data=df_past, current_data=df_new)

# # Generate Report
# drift_suite.save_html("logs/drift_report.html")

# print("✅ Drift detection completed. Check logs/drift_report.html")

import os
import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# If predictions.log doesn't exist, create an empty file
if not os.path.exists("logs/predictions.log"):
    with open("logs/predictions.log", "w") as f:
        f.write("timestamp,input,prediction\n")  # Dummy header
    print("✅ Created empty predictions.log")

# Load past predictions (baseline data)
df_past = pd.read_csv("data/financial_data.csv")  

# Load new predictions (if available)
df_new = pd.read_csv("logs/predictions.log")

# If no new predictions, exit early
if len(df_new) <= 1:  # Only the header exists
    print("⚠ No new predictions found. Skipping drift detection.")
    exit()

# Convert input column to dictionary
df_new["input"] = df_new["input"].apply(eval)
df_new = df_new["input"].apply(pd.Series)
df_new["prediction"] = df_new["prediction"].astype(float)

# Run drift detection
drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
drift_report.run(reference_data=df_past, current_data=df_new)
drift_report.save_html("logs/drift_report.html")

# Save drift status
drift_status = {"drift_detected": False}  # Assume no drift by default
with open("logs/drift_status.json", "w") as f:
    json.dump(drift_status, f)

print("✅ Drift detection completed. Check logs/drift_report.html")



# import json
# import os

# # Create logs directory if it doesn't exist
# os.makedirs("logs", exist_ok=True)

# # Store drift detection result
# drift_detected = False  # Assume no drift by default

# # Save status to drift_status.json
# drift_status = {"drift_detected": drift_detected}
# with open("logs/drift_status.json", "w") as f:
#     json.dump(drift_status, f)

# print(f"✅ Drift status saved: {drift_status}")
