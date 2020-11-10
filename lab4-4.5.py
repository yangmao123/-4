# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 05:37:33 2020

@author: 75965
"""

from matplotlib.pyplot import boxplot
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans

#Train data 模型
FILE_PATH = 'tic-tac-toe.txt'
df = pd.read_csv(FILE_PATH)
source = np.array(df)
NumDataPerClass=200
X = source[:,[0,1,2,3,4,5,6,7,8]]
y = source[:,[9]]
rIndex = np.random.permutation(2*NumDataPerClass)
Xr = X[rIndex,]
yr = y[rIndex]
# Training and test sets (half half)
#
X_train = Xr[0:NumDataPerClass]
y_train = yr[0:NumDataPerClass]
X_test = Xr[NumDataPerClass:2*NumDataPerClass]
y_test = yr[NumDataPerClass:2*NumDataPerClass]
Ntrain = NumDataPerClass;
Ntest = NumDataPerClass;
w = np.random.randn(9)
MaxIter=2500
alpha = 0.002
# Space to save answers for plotting
#
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)
# Main Loop
#
for iter in range(MaxIter):
# Select a data item at random
#
    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]
# If it is misclassified, update weights
#
    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x
# Evaluate trainign and test performances for plotting
#
#1和2.1和2.2
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
def gaussian(x, u, sigma):
  return(np.exp(-0.5 * np.linalg.norm(x-u) / sigma))
N, p = X.shape

# Space for design matrix
#
M = 200
U = np.zeros((N,M))
# Basis function locations at random
#
C = np.random.randn(M,p)
# Basis function range as distance between two random data
#
x1 = X[np.floor(np.random.rand()*N).astype(int),:]
x2 = X[np.floor(np.random.rand()*N).astype(int),:]
sigma = np.linalg.norm(x1-x2)
# Construct the design matrix
#
for i in range(N):
  for j in range(M):
    U[i,j] = gaussian(X[i,:], C[j,:], sigma)
# Pseudo inverse solution for linear part
#
l = np.linalg.inv(U.T @ U) @ U.T @ y
yh = U @ l

#2-(1)
#Normalize data
X_scaled = preprocessing.scale(X)
#2-(1)
#2-(2)
#Sigma based on many distance 
#设置的数值越大，噪音越小
normalize_sigma = 50
sigmas = [ ] 
for _ in range(normalize_sigma):
    x1 = X[np.floor(np.random.rand()*N).astype(int),:]
    x2 = X[np.floor(np.random.rand()*N).astype(int),:]
    sigmas.append(np.linalg.norm(x1-x2))
sigma_many = np.mean(sigmas)

#2-(2)
for i in range(N):
    for j in range(M):
        U[i,j] = gaussian(X[i,:], C[j,:], sigma)
# Pseudo inverse solution for linear part
#
l = np.linalg.inv(U.T @ U) @ U.T @ y
# Predicted values on training data
#

#2.3
kmeans = KMeans(M) 
kmeans.fit(X_scaled)
C_kmeans = kmeans.cluster_centers_

#2.4
X_train, X_test,y_train, y_test = train_test_split(X_scaled, y, test_size=0.1,random_state=5)
N_train, p_train = len(X_train),len(y_train)
N_test, p_test = len(X_test), len(y_test)

U_train = np.zeros((N_train, M))
for i in range(N_train):
    for j in range(M):
        U_train[i,j] = gaussian(X_train[i,:],C_kmeans[j,:],sigma_many)

#pseudo inverse solution for linear part on train data
l_train =np.linalg.inv(U_train.T @ U_train) @ U_train.T @ y_train
#construct a design matrix for test data
U_test = np.zeros((N_test,M))
for i in range(N_test):
    for j in range(M):
        U_test[i,j] = gaussian(X_test[i,:],C_kmeans[j,:],sigma_many)

#predicted  values on training,test data
yh_train = U_train @ l_train
yh_test = U_test @ l_train
#plotting
yh = U @ l

def error(t, t_hat):
    return((t - t_hat)**2).mean()
train_err = error(y_train, yh_train)
test_err = error(y_test, yh_test)


fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(9,3))
axes[0].boxplot(yh_train)
axes[1].boxplot(yh_test)
axes[2].boxplot(yh_test)
axes[2].set_ylim(0,1000)