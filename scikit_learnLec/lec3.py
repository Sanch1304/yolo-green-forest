
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()
print(bc)

X = scale(bc.data)
print(X)

y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

predictions = model.predict(X_test)

labels = model.labels_

print("labels: ", labels)
print("Predictions: ", predictions)
print("accuracy: ", accuracy_score(y_test, predictions))
print("Actual: ", y_test)
print(pd.crosstab(y_train, labels))



