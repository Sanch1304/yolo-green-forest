# knn 


import numpy as np
import pandas as pd
from sklearn import neighbors, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

data = pd.read_csv('car.data')
print(data.head())

X = data[['buying', 'maint', 'safety']].values
y = data[['class']]
X = np.array(X)
print(X)

Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
print(X)
# file transform for y is not used To avoid potential errors caused by automatic or unexpected reordering of labels
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)

#create model
print(X.shape)
print(y.shape)

knn = svm.SVC()
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions:", prediction)
print("accuracy: ", accuracy)
