import numpy as np
import scipy.stats as sp
import scipy
import matplotlib.pyplot as plt
import subprocess
import csv

def pdf(mean, var, x):
   return np.divide(1,float((np.sqrt(2*np.pi*var))) * np.exp(np.divide((float((mean - x)**2)), 2*var)))

X = []
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

##print(Y)
data_range = len(Y)

X00 = []
X01 = []
X10 = []
X11 = []

count1 = 0
count0 = 0

for i in range(0,data_range):

    if(Y[i] == 1):
        #print('working')
        count1+=1
        X10.append((X[0][i]))
        X11.append((X[1][i]))
    else:
        count0+=1
        X00.append((X[0][i]))
        X01.append((X[1][i]))
#print(X00)

## mean and variance for variable 1 and X[0]
mean10 = np.mean(X10)
var10  = np.var(X10)

## mean and variance for variable 1 and X[1]
mean11 = np.mean(X11)
var11 = np.var(X11)

## mean and variance for variable 0 and X[0]
mean00 = np.mean(X00)
var00 = np.var(X00)

## mean and variance for variable 0 and X[1]
mean01 = np.mean(X01)
var01 = np.var(X01)

input1 = [1,1]
input2 = [1,-1]
input3 = [-1,1]
input4 = [-1,-1]

print(count1 )
print(count0)
probY1 = float(count1)/1000
probY0 = float(count0)/1000

# Label 0, mean X0

print('probY1 = ', probY1)
print('probY0 = ', probY0)

pdf_input10_label1 = pdf(mean10, var10, input1[0])
pdf_input11_label1 = pdf(mean11, var11, input1[1])
pdf_input1_1 = pdf_input10_label1 * pdf_input11_label1 * probY1


pdf_input10_label0 = pdf(mean00, var00, input1[0])
pdf_input11_label0 = pdf(mean01, var01, input1[1])
pdf_input1_0 = pdf_input10_label0 * pdf_input11_label0 * probY0

if(pdf_input1_1 > pdf_input1_0):
 print('for input [1,1] = 1')
else:
 print('for input [1,1] = -1')
################################################################################

pdf_input20_Y1 = pdf(mean10, var10, input2[0])
pdf_input21_Y1 = pdf(mean11, var11, input2[1])
pdf_input2_Y1 = pdf_input20_Y1 * pdf_input21_Y1 * probY1

pdf_input20_Y0 = pdf(mean00, var00, input2[0])
pdf_input21_Y0 = pdf(mean01, var01, input2[1])
pdf_input2_Y0 = pdf_input20_Y0 * pdf_input21_Y0 * probY0

if(pdf_input2_Y1 > pdf_input2_Y0):
    print('pdf [1,-1] = 1')
else:
    print('pdf [1,-1] = -1')

################################################################################

pdf_input30_label1 = pdf(mean10, var10, input3[0])
pdf_input31_label1 = pdf(mean11, var11, input3[1])
pdf_input3_1 = pdf_input30_label1 * pdf_input31_label1 * probY1


pdf_input30_label0 = pdf(mean00, var00, input3[0])
pdf_input31_label0 = pdf(mean01, var01, input3[1])
pdf_input3_0 = pdf_input30_label0 * pdf_input31_label0 * probY0

if(pdf_input3_1 > pdf_input3_0):
 print('for input [-1,1] = 1')
else:
 print('for input [-1,1] = -1')

################################################################################


pdf_input40_label1 = pdf(mean10, var10, input4[0])
pdf_input41_label1 = pdf(mean11, var11, input4[1])
pdf_input4_1 = pdf_input40_label1 * pdf_input41_label1 * probY1


pdf_input40_label0 = pdf(mean00, var00, input4[0])
pdf_input41_label0 = pdf(mean01, var01, input4[1])
pdf_input4_0 = pdf_input40_label0 * pdf_input41_label0 * probY0

if(pdf_input4_1 > pdf_input4_0):
 print('for input [-1,-1] = 1')
else:
 print('for input [-1,-1] = -1')
