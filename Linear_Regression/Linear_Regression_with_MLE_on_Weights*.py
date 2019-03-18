import numpy as np
#import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess



## Question 4


################################################################################
#no of training examples

N = input('number of training examples  ')
M = input('number of testing examples   ')

################################################################################
############### Training Data
Xl = np.linspace(0, 2*np.pi , N)   ## generating initial X column matrix
Y2 = np.sin(Xl)
Y2 += np.random.normal(0 , 0.05 , N)  ## adding noise to variable matrix
Xl = np.reshape(Xl,(N,1))

Bias = np.ones((N,1))

Xl = np.hstack((Bias , Xl))  ## adding bias term to the input matrix

Y = np.random.normal(Y2,0.05,N)

Y_std = np.std(Y)

Y_mean = np.mean(Y)
Y_mean = np.ones((M,1)) * Y_mean

Y = np.reshape(Y, (N,1))
W = np.matmul(np.linalg.pinv(Xl), Y)

################################################################################
############### Testing Data

X_test = np.linspace(0 , 2*np.pi , M)
X_test = np.reshape(X_test, (M,1))

Bias2 = np.ones((M,1))   ## adding bias term to test input matrix
X_test = np.hstack( (Bias2, X_test))

Y_test = np.matmul(X_test, W)# - Y_mean



YT_var = np.std(Y_test-Y)
#print('assumed deviation = ', Y_dev)
print('assumed deviation for training data = ', Y_std)
print('deviation for test data =  ', YT_var)

################################################################################
## Estimated variables as input Plotting
Y_test = np.reshape(Y_test, (1,M))
Y_test += np.random.normal(0 , 0.05 , M)
Y_test = np.reshape(Y_test, (M,1))
W1 = np.matmul(np.linalg.pinv(X_test), Y_test)

X_test2 = np.linspace(0 , 2*np.pi , M)
X_test2 = np.reshape(X_test2, (M,1))

Bias3 = np.ones((M,1))   ## adding bias term to test input matrix
X_test2 = np.hstack( (Bias2, X_test2))

Y_test2 = np.matmul(X_test2, W1)# - Y_mean



################################################################################
############### setting up axes

X_axisN = np.linspace(0, 2*np.pi, N)
X_axisN = np.reshape(X_axisN, (N,1))

X_axisM = np.linspace(0, 2*np.pi, M)
X_axisM = np.reshape(X_axisM, (M,1))

################################################################################
############### Plotting

plt.plot(X_axisN , Y, 'o')
plt.plot(X_axisM, Y_test2)
plt.plot(X_axisM, Y_test)

plt.grid()
plt.xlabel('0')
plt.ylabel('sinusoidal value')
plt.legend(["Training Data" , "Testing Data", "max likelihood labels"])
plt.show()
