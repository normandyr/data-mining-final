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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sns.set_style("white")



data = pd.read_csv('divorce.csv')
X = np.array(data.iloc[:, 1:])
y = np.array(data["Class"])

"Using KNN on categorical data"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

sns.catplot(x = 'Class', kind='count', palette="ch:.25", data = data)


#data.info()


