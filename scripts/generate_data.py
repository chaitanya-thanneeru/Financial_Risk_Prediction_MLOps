import pandas as pd
import numpy as np

# Generate synthetic financial data
np.random.seed(42)
data = {
    "market_volatility": np.random.rand(100) * 10,
    "interest_rate": np.random.rand(100) * 5,
    "inflation_rate": np.random.rand(100) * 3,
    "stock_price_change": np.random.rand(100) * 15,
    "risk_score": np.random.randint(1, 100, 100)  # Target variable
}

df = pd.DataFrame(data)
df.to_csv("data/financial_data.csv", index=False)
print("✅ Financial dataset saved successfully!")
