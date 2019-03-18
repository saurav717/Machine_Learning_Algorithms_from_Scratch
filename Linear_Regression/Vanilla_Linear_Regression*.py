import numpy as np
#import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess

#no of training examples

################################################################################

N = input('number of training examples  ')
M = input('number of testing examples   ')


## Question 1

## Vanilla Regression


################################################################################
############### Training Data
Xl = np.linspace(1, 2*np.pi , N)
Y = np.sin(Xl)
Y += np.random.normal(0 , 0.05 , N)

Xl = np.reshape(Xl,(N,1))
Bias = np.ones((N,1))

Xl = np.hstack((Bias, Xl))


Y = np.reshape(Y, (N,1))
W = np.matmul(np.linalg.pinv(Xl), Y)

print (W)
################################################################################
############### Testing Data

X_test = np.linspace(1 , 2*np.pi , M)
X_test = np.reshape(X_test, (M,1))

Bias2 = np.ones((M,1))

X_test = np.hstack((Bias2, X_test))
Y_test = np.matmul(X_test, W)


################################################################################
############### setting up axes

X_axisN = np.linspace(0, 2*np.pi, N)
X_axisN = np.reshape(X_axisN, (N,1))

X_axisM = np.linspace(0, 2*np.pi, M)
X_axisM = np.reshape(X_axisM, (M,1))

################################################################################
############### Plotting

plt.plot(X_axisN , Y, 'o')
plt.plot(X_axisM, Y_test)
plt.grid()
plt.xlabel('0')
plt.ylabel('sinusoidal value')
plt.legend(["Training Data" , "Testing Data", "Vanilla Regression"])
plt.show()
