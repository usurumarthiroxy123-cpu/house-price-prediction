import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = {
    "Area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2300],
    "Price": [1500000, 2400000, 3000000, 3600000, 4500000, 5400000, 6000000, 6900000]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Split Data
# -----------------------------
X = df[["Area"]]   # Independent variable
y = df["Price"]    # Dependent variable

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Step 4: Prediction
# -----------------------------
area = int(input("Enter house area in sq ft: "))
predicted_price = model.predict([[area]])

print(f"Predicted House Price: ₹ {int(predicted_price[0])}")

# -----------------------------
# Step 5: Visualization
# -----------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
