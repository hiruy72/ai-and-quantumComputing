import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data: House sizes (sq ft) and corresponding prices
X = np.array([500, 700, 900, 1100, 1300]).reshape(-1, 1)  # Feature (sq ft)
y = np.array([150000, 180000, 210000, 250000, 280000])  # Target (price)

# Create Linear Regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict price for a house with 1000 sq ft
predicted_price = model.predict([[1000]])
print(f"Predicted Price for 1000 sq ft house: ${predicted_price[0]:,.2f}")

# Plot Data and Regression Line
plt.scatter(X, y, color='blue', label="Actual Prices")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()