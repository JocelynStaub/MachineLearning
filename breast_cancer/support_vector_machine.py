# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:54:36 2017

@author: xILENCE
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm

df = pd.read_csv('breast-cancer-wisconsin.data.txt', names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','class'])
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

clf = svm.SVC()
#clf = neighbors.KNeighborsClassifier()    #for KNN
clf.fit(X_train,y_train)

accuracy = clf.score(X_test, y_test)

print('Accuracy :',accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print('Prediction :',prediction)