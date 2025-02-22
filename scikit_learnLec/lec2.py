from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('Boston.csv')
print(df)

X = df.drop('medv',axis=1)
X = df.drop('sr',axis=1).values
y = df['medv'].values
#
print("X")
print(X)
print(X.shape)
print("y")
print(y)
print(y.shape)

#algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train
model = l_reg.fit(X_train, y_train)

predictions = model.predict(X_test)
# acc = accuracy_score(predictions,y_test) in linear regression accuracy is not used
print("predictions: ", predictions)
print("R^2: ", l_reg.score(X, y))
print("coeff: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)