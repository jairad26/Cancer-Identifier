import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection, neighbors #model_selection replaced cross validation
from sklearn.model_selection import cross_validate, train_test_split #just in case, I imported these as well
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

df=pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)   #most algorithms can see outliers and know not to use it
#df.dropna(inplace=True) #could do this as well
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,1,3,3,2,4]])

example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)


