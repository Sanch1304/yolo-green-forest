# pipelines

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv('train.csv')
print(df.head())
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

X_train , X_test ,y_train , y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2)


# Impute transformer
trf1 = ColumnTransformer(
    [('impute_age', SimpleImputer(), [2]),
     ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])],
    remainder='passthrough'
)
#one hot encoder
trf2 = ColumnTransformer(
    [('ohe_sex_embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])]
    ,remainder='passthrough'
)

##Scaling
trf3 = ColumnTransformer(
    [('scale',MinMaxScaler(),slice(0,8))]
)

#Feature Selection

trf4 = SelectKBest(score_func=chi2,k=5)

trf5 = DecisionTreeClassifier()

pipe = Pipeline(
    [
        ('trf1',trf1),
        ('trf2',trf2),
        ('trf3',trf3),
        ('trf4',trf4),
        ('trf5',trf5)
    ]
)

model = pipe.fit(X_train,y_train)
print(X_test)
prediction = model.predict(X_test)
print(prediction)

with open('model.pkl','wb') as f:
    pickle.dump(model,f)

with open('model.pkl','rb') as f:
    model1 = pickle.load(f)


test_input = np.array([[2,'male',31.0,0,0,10.5,'S'],[2,'female',31.0,0,0,10.5,'S']],dtype=object).reshape(2,7)

prediction = model1.predict(test_input)
print(prediction)










































































































































































































