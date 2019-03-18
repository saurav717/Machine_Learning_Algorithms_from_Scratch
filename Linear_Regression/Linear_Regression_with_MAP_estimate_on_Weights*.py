import numpy as np
#import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess


## Question 5

N = input('enter number of training data     ')
d = input('enter degree of polynomial    ')
M = input('enter number of testing data    ')

sigma = input('enter the standard deviation of test variables   ')
alpha = input('enter the standard deviation of parameters   ')


#################################################################################
######################### Training Data
Xl = np.linspace(1, 2*np.pi, N)
Y = np.sin(Xl)
Y += np.random.normal(0 , 0.05 , N) ## adding noise to sinusoidal labels
Xl = np.reshape(Xl, (N,1))          ## converting X matrix to column matrix
Y = np.reshape(Y, (N,1)) - np.mean(Y)

I = np.identity(d)

X = Xl**0

for i in range(1,d):           # for making N*d dimensional polynomial input X matrix
    X = np.hstack((X,Xl**i))

l = np.divide(np.square(sigma) , np.square(alpha))                ## L = ((sigma)^2) / ((lambda)^2)

W = np.matmul(np.linalg.inv(np.matmul(X.T,X) + l*I), np.matmul(X.T,Y))  ## W = ((X'X)^-1) * X'Y

#################################################################################
########################## Testing Data

X_tes = np.linspace(1,2*np.pi, M)
X_tes = np.reshape(X_tes, (M,1))

X_test = X_tes**0

for j in range(1,d):                 ## for making M*d dimensional polynomial test X matrix
    X_test = np.hstack((X_test, X_tes**j))

Y_test = np.matmul(X_test,W) + np.mean(Y)

X_axisM = np.linspace(1,2*np.pi, M)    ## fixing the axes
X_axisM = np.reshape(X_axisM, (M,1))

X_axisN = np.linspace(1,2*np.pi, N)    ## fixing the axes
X_axisN = np.reshape(X_axisN, (N,1))

#################################################################################
########################### Plotting Part

plt.plot(X_axisN, Y, 'bo')
plt.plot(X_axisM, Y_test, 'ro')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(["training data", "testing data" ])
plt.show()
