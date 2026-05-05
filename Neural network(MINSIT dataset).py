from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

#PART A: Dataset exploration

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

#A1: shape, sample and features
print("Shape:", X.shape)

# A2: Display 10 images
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(y[i])
    plt.axis('off')
plt.show()

#A3: Class distribution
sns.countplot(x=y)
plt.show()

#PART B: Data Analysis
print("Min:", X.min().min())
print("Max:", X.max().max())
print("Mean:", X.mean().mean())

X = X / 255.0
#PART C: Model implementation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_acc = lr.score(X_test, y_test)
print("Logistic Regression Accuracy:", lr_acc)

nn = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20)
nn.fit(X_train, y_train)

nn_acc = nn.score(X_test, y_test)
print("Neural Network Accuracy:", nn_acc)

#PART E: Experimentation
nn = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20)
nn.fit(X_train, y_train)

nn_acc = nn.score(X_test, y_test)
print("Neural Network Accuracy:", nn_acc)

# PART F: PCA
for n in [20, 50, 100]:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_pca, y_train_pca)

    print(f"PCA {n} Accuracy:", lr.score(X_test_pca, y_test_pca))

# PART G: Misclassificatoin + Confusion matrix
y_pred = nn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
mis = np.where(y_pred != y_test)[0]

for i in mis[:5]:
    plt.imshow(X_test.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}, True: {y_test.iloc[i]}")
    plt.axis('off')
    plt.show()

