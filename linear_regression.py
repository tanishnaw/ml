import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x=np.array([1,2,3,4,5]).reshape(-1, 1)
y=np.array([50, 55, 65, 70, 80])
X_b = np.c_[np.ones((x.shape[0], 1)), x]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
y_pred_normal = X_b @ theta
mse_normal = mean_squared_error(y, y_pred_normal)
print("Normal Equation Results:")
print("Intercept:", theta[0])
print("Slope:", theta[1])
print("MSE:", mse_normal)
model = LinearRegression()
model.fit(x, y)
y_pred_sklearn = model.predict(x)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
print("\nSklearn LinearRegression Results:")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("MSE:", mse_sklearn)
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred_normal, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.title("Linear Regression using Least Squares")
plt.legend()
plt.show()