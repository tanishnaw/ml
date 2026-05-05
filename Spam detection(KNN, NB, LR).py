from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = fetch_20newsgroups(subset='train')
X = data.data
y = data.target
vectorizer = TfidfVectorizer(stop_words='english')
X_transformed = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

#Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_acc = nb_model.score(X_test, y_test)
print("Naive Bayes Accuracy:", nb_acc)

#KNN (try multiple K)
print("\nKNN Results:")
for k in [3, 5, 7]:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_acc = knn_model.score(X_test, y_test)
    print(f"K={k}, Accuracy={knn_acc}")

#Logistic Regression (L2)
lr_model_l2 = LogisticRegression(penalty='l2', max_iter=200)
lr_model_l2.fit(X_train, y_train)
lr_acc_l2 = lr_model_l2.score(X_test, y_test)
print("\nLogistic Regression (L2) Accuracy:", lr_acc_l2)

#Logistic Regression (L1)
lr_model_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
lr_model_l1.fit(X_train, y_train)
lr_acc_l1 = lr_model_l1.score(X_test, y_test)
print("Logistic Regression (L1) Accuracy:", lr_acc_l1)

