import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([
    [800, 2, 20],
    [1000, 2, 15],
    [1200, 3, 18],
    [1500, 3, 10],
    [1800, 4, 8],
    [2000, 4, 5],
    [2200, 5, 7],
    [2500, 5, 3]
], dtype=float)

y = np.array([50, 60, 72, 90, 115, 130, 150, 180], dtype=float)

# (A) Train-Test Split (10%) & Min-Max Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (B) Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# (C) Polynomial Regression (Degree = 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)

y_pred_pr = pr.predict(X_test_poly)

mse_pr = mean_squared_error(y_test, y_pred_pr)
r2_pr = r2_score(y_test, y_pred_pr)

# (E) Prediction for New House
new_house = np.array([[2100, 4, 6]])
new_house_scaled = scaler.transform(new_house)

price_lr = lr.predict(new_house_scaled)
price_pr = pr.predict(poly.transform(new_house_scaled))

# Results
print("Linear Regression")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

print("\nPolynomial Regression (Degree 2)")
print("MSE:", mse_pr)
print("R2 Score:", r2_pr)

print("\nPredicted Price for Area=2100, Bedrooms=4, Age=6")
print("Linear Regression:", price_lr[0])
print("Polynomial Regression:", price_pr[0])

# (F) Plot Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred_lr, label="Linear Regression")
plt.scatter(y_test, y_pred_pr, label="Polynomial Regression")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
