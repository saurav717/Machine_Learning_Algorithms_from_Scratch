import numpy as np
import scipy.stats as sp
import scipy
#import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import cvxpy as cp
import csv

X = []
Y = []

with open('Xsvm.csv')as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        row = list(map(float,row))
        X.append(row)

with open('ysvm.csv')as csvfile2:
    spamreader2 = csv.reader(csvfile2)
    for column in spamreader2:
        Y.append(int(float(column[0])))

a = Alpha = cp.Variable(len(Y))
X = np.reshape(X,(len(Y),2))
Y = np.reshape(Y,(len(Y),1))

Alpha = cp.diag(Alpha)
AY = cp.matmul(Alpha,Y)
AYX = cp.matmul(cp.diag(AY),X)
prob = cp.norm(AYX)**2

prob1 = cp.Maximize(cp.sum(Alpha) - 0.5*prob)
constraints = [a>=0,  cp.matmul(Alpha.T ,Y)==0]
prob2 = cp.Problem(prob1, constraints)
prob2.solve(verbose = True)
#print(Alpha)
Al = (np.array(a.value)).reshape(len(Y),1)
W = np.zeros((2,))
for i in range((len(Y))):
    W = W + Al[i]*Y[i]*(X[i].T)
    if(Al[i]>1e-4):
        print(i)
print(W)
W0 = (1/Y[281]) - np.dot(W,X[281])
print("W0 = ",W0)
test = np.array([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008,0.001]])
for i in range(len(test)):
    E = np.sign(np.dot(W,test[i])+W0)
    print(test[i], W)

from sklearn import svm
clf = svm.SVC()
clf.fit(X,Y)
clf.predict(test)
