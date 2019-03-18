import numpy as np
#import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess

##Question 2

## estimating using polynomial function of degree d

################################################################################
#number of examples
N = input('enter number of training data     ')
d = input('enter degree of polynomial    ')
M = input('enter number of testing data    ')


#################################################################################
######################### Training Data
Xl = np.linspace(1, 2*np.pi, N)

Y = np.sin(Xl)
Y += np.random.normal(0 , 0.05 , N)
Xl = np.reshape(Xl, (N,1))

Y = np.reshape(Y, (N,1))

X = Xl**0

for i in range(1,d):
    X = np.hstack((X,Xl**i))

W = np.matmul(np.linalg.pinv(X), Y)

#################################################################################
########################## Testing Data

X_tes = np.linspace(1,2*np.pi, M)
X_tes = np.reshape(X_tes, (M,1))

X_test = X_tes**0

for j in range(1,d):
    X_test = np.hstack((X_test, X_tes**j))

Y_test = np.matmul(X_test,W)

X_axisM = np.linspace(1,2*np.pi, M)
X_axisM = np.reshape(X_axisM, (M,1))

X_axisN = np.linspace(1,2*np.pi, N)
X_axisN = np.reshape(X_axisN, (N,1))

#################################################################################
########################### Plotting Part

plt.plot(X_axisN, Y, 'o')
plt.plot(X_axisM, Y_test)
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(["training data", "testing data" ])
plt.show()
