import numpy as np
import scipy.stats as sp
import scipy
import matplotlib.pyplot as plt
import subprocess
import csv

def bubblesort(l,y):
    for i in range (len(l)-1,0,-1):
        for j in range(i):
            if(l[j]>l[j+1]):
                l[j],l[j+1] = l[j+1],l[j]
                y[j],y[j+1] = y[j+1], y[j]
                return(l,y)

pp1 = [10,5,4,0,12,34]
pp2 = [0,1,0,1,0,1]


def dist(X,Y):
    return (np.sqrt((X[0] - Y[0])**2 + (X[1] - Y[1])**2))

X = []

k = input('enter k values')

with open('X.csv') as csvfile:
     spamreader = csv.reader(csvfile)
     for row in spamreader:
         row = list(map(float,row))
         X.append(row)

Y = []
with open('Y.csv') as csvfile2:
    spamreader2 = csv.reader(csvfile2)
    for column in spamreader2:
         Y.append(int(float((column[0]))))

X = np.column_stack((X[0],X[1]))


X_train = X[:800]
X_test = X[:-800]

#print(len(X))


Y_train = Y[:800]
Y_test = Y[:-800]

Dist = []
ix = 0
for i in X_test:
    D = []
    #print("count" + str(ix))
    for j in range(len(X_train)):
            D.append(dist(X_train[j],i))
            #print(D)
    c1 = bubblesort(D,Y_train)
    #print(k)
    c1 = c1[:k]
    cn1 = c1.count(1)
    print(cn1)
    cn2 = c1.count(-1)
    print(cn2)
    if(cn1>cn2):
        Dist.append(1)
    else:
        Dist.append(-1)
    ix+=1



print(len(Dist))
print(Dist)

xx = []
for i in range(len(Dist)):
    xx.append(float(Dist[i])+Y_test[i])

xxx = xx.count(0)
xxx
print((float(xxx)/len(xx))*100)
