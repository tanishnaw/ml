from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

print("Train-Test Results:")
for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k}, Accuracy={acc:.4f}")

print("\nCross-Validation Results:")
for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_transformed, y, cv=5)
    print(f"k={k}, Cross-Val Accuracy={scores.mean():.4f}")

    