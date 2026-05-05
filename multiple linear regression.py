import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Independent variables: [Study Hours, Sleep Hours]
X = np.array([
    [1, 6],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8]
])

# Dependent variable: Exam Score
y = np.array([50, 55, 65, 70, 80])

# -------------------------------
# 1. Normal Equation Method
# -------------------------------
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

y_pred_normal = X_b @ theta
mse_normal = mean_squared_error(y, y_pred_normal)

print("Normal Equation Results:")
print("Intercept:", theta[0])
print("Coefficients:", theta[1:])
print("MSE:", mse_normal)

# -------------------------------
# 2. Scikit-learn Method
# -------------------------------
model = LinearRegression()
model.fit(X, y)

y_pred_sklearn = model.predict(X)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print("\nSklearn LinearRegression Results:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("MSE:", mse_sklearn)

# -------------------------------
# Visualization (only 1 feature at a time)
# -------------------------------
plt.scatter(X[:, 0], y, label="Study Hours vs Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Multiple Linear Regression (2 Features)")
plt.legend()
plt.show()
