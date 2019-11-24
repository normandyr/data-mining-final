# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:54:32 2019

@author: N
"""

'''import os
dirpath = os.getcwd()
print("current directory is : " + dirpath)

os.chdir("\\Users\\A\\Desktop")

dirpath = os.getcwd()
print("current directory is : " + dirpath)'''

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

#sns.set_style("white")

#f, ax = plt.subplots(figsize=(30, 30))

#here we have our usable dataset with X corresponding to the attributes and y corresponding to the class outcome
data = pd.read_csv('transfusion.csv')
print(data)
X = np.array(data[["Months", "Times", "Blood", "Frequency"]])
print(X)
y = np.array(data["Donated"])

'''this doesn't help seems to cause everything to overfit'''
#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)

'''using MinMaxScaler with shuffle doesn't change it'''
'''using shuffle by itself seems to lower accuracy'''
#seed = 42
#X, y = shuffle(X, y, random_state=seed)
'''Can alter the Naive Bayes by showing the overfitting more if change test size to 0.99 which will result in the print(clf.class_prior_) equally the distribution of the dominant class label at .76 or around this
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.99, random_state=0)'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state=0)

sns.countplot(x='Donated', data=data)

###SVM not appropriate as our output data is not continous
clf = LinearSVC(random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X_test)
print("SVM ", accuracy_score(y_test, y_pred))





logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))



clf = GaussianNB()
clf.fit(X_train, y_train)
print(clf.class_prior_)
print(clf.get_params)
y_pred = clf.predict(X_test)


#print("Accuracy:", accuracy_score(y_test, y_pred))
print("Naive Bayes ", clf.score(X_test, y_test))
print("Naive Bayes ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

for k in range(1, 27):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("knn accuracy score ", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))







