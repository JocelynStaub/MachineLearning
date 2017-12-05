# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:44:48 2017

@author: xILENCE
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm    #support vector machine

digits = datasets.load_digits()

#print(digits.target)
#print(digits.data)
#print(digits.images[0])

clf = svm.SVC(gamma = 0.001, C = 100)
#print(len(digits.data))
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)
index = 1  #select which sample to look at
print('Prediction :', clf.predict(digits.data[index].reshape(1, -1)))
plt.imshow(digits.images[index], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()