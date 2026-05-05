import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [800, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2200, 4]
], dtype=float)

y = np.array([50, 75, 90, 115, 140], dtype=float).reshape(-1, 1)

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def gradient_descent(X, y, lr=0.01, iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n + 1, 1))
    cost_history = []
    
    for i in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        gradient = (2/m) * X_b.T.dot(errors)
        theta = theta - lr * gradient
        cost = compute_mse(y, predictions)
        cost_history.append(cost)
        
    return theta, cost_history

theta_raw, cost_raw = gradient_descent(X, y, lr=0.01, iterations=1000)
X_b_raw = np.c_[np.ones((X.shape[0], 1)), X]
y_pred_raw = X_b_raw.dot(theta_raw)

X_min, X_max = X.min(axis=0), X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)
theta_scaled, cost_scaled = gradient_descent(X_scaled, y, lr=0.01, iterations=1000)
X_b_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
y_pred_scaled = X_b_scaled.dot(theta_scaled)

print("--- Model Performance ---")
print(f"Unscaled - MSE: {compute_mse(y, y_pred_raw):.2f}, R2: {compute_r2(y, y_pred_raw):.4f}")
print(f"Scaled   - MSE: {compute_mse(y, y_pred_scaled):.2f}, R2: {compute_r2(y, y_pred_scaled):.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cost_raw)
plt.title('Cost vs Iterations (No Scaling)')
plt.ylabel('Cost (MSE)')
plt.subplot(1, 2, 2)
plt.plot(cost_scaled)
plt.title('Cost vs Iterations (Min-Max Scaling)')
plt.ylabel('Cost (MSE)')
plt.show()
