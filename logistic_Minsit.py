# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:32:33 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P


def getTransformedData():
  print('Reading in and transforming data')
  
  df=pd.read_csv('')
  data=df.as_matrix().astype(np.float32)
  np.random.shuffle(data)
  X=data[:,1:]
  mu=X.mean(axis=0)
  X=X-mu
  pca=PCA()
  Z=pca.fit_transform(X)
  Y=data[:,0].astype(np.int32)
  
  plot_cumulative_variance(pca)
  
  return Z,Y,pca,mu
  
def getNormalizedData():
  print("Reading in and tranforming data")
  
  df=pd.read_csv('E:/RS/ML/Machine learning tuts/Target/Part2(Deep Learning)/DeepLearning-Part2/train.csv')
  data=df.as_matrix().astype(np.float32)
  np.random.shuffle(data)
  X=data[:,1:]
  mu=X.mean(axis=0)
  std=X.std(axis=0)
  np.place(std,std==0,1)
  X=(X-mu)/std
  Y=data[:,0]
  return X, Y
  
def forward(X,W,b):
  Z=X.dot(W)+b
  expA=np.exp(Z)
  return expA/expA.sum(axis=1,keepdims=True)

def errorRate(Y,T):
  return np.mean(Y!=T)
  
def cost(Y,T):
  tot = T*np.log(Y)
  return -tot.sum()
  
def gradW(Y,T,X):
  return X.T.dot(Y-T)
  
def gradb(Y,T):
    return (Y-T).sum(axis=0)
  
def y2indicator(Y):
  N,M=len(Y),len(set(Y))
  Yind=np.zeros((N,M))
  for i in range(N):
    Yind[i,Y[i]]=1
  return Yind
  
def benchmarkFull():
  X,Y=getNormalizedData()
  print("Performing LogisticRegression")
   
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)
  
  N,D=Xtrain.shape
  K=10
  W=np.random.randn(D,K)/np.sqrt(D)
  b=np.zeros(K)
  lr=0.00004
  reg=0.01
  costs=[]
  for i in range(500):
    pY=forward(Xtrain,W,b)
    W-=lr*(gradW(pY,Ytrain_ind,Xtrain)+reg*W)
    b-=lr*(gradb(pY,Ytrain_ind)+reg*b)
    if i%10==0:
      pYtest=forward(Xtest,W,b)
      c=cost(pYtest,Ytest_ind)
      costs.append(c)
      e=errorRate(np.argmax(pYtest,axis=1),Ytest)
      print("At iteration i=%d cost is %.6f"%(i,c))
      print("Error Rate is",e)
    plt.plot(costs)
    
def benchmarkPca():
  X,Y,_,_=getTransformedData()
  X=X[:,:300]
  X=X-X.mean()
  X=X/X.std()
  
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]

  N,D=Xtrain.shape
  K=10
  
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)
  W=np.random.randn(D,K)/np.sqrt(D)
  b=np.zeros(K)
  lr = 0.0001
  reg = 0.01
  costs=[]
  for i in range(500):
    pY=forward(Xtrain,W,b)
    W+=lr*(gradW(pY,Ytrain_ind,Xtrain)-reg*W)
    b+=lr*(gradb(pY,Ytrain_ind)-reg*b)
    if i%10==0:
      pYtest=forward(Xtest,W,b)
      c=cost(pYtest,Ytest_ind)
      costs.append(c)
      e=errorRate(np.argmax(pYtest,axis=1),Ytest)
      print("At iteration i=%d cost is %.6f"%(i,c))
      print("Error Rate is",e)
  plt.plot(costs)
  
  
      
if __name__=='__main__':
  benchmarkPca()
  #benchmarkFull()
  