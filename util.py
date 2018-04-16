# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:57:59 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plotCumulativeVariance(pca):  
  P=[]
  for p in pca.explained_variance_ratio_:
    if len(P)==0:
      P.append(p)
    else:
      P.append(p+P[-1])
  plt.plot(P)
  return P

def getTransformedData():
  df=pd.read_csv('E:/RS/ML/Machine learning tuts/Target/Part2(Deep Learning)/DeepLearning-Part2/train.csv')
  data=df.as_matrix().astype(np.float32)
  np.random.shuffle(data)
  X=data[:,1:]
  mu=X.mean(axis=0)
  X=X-mu#center the data
  pca=PCA()
  Z=pca.fit_transform(X)
  Y=data[:,0].astype(np.int32)
  plotCumulativeVariance(pca)
  
  return Z,Y,pca,mu

def getNormalizedData():
  df=pd.read_csv('E:/RS/ML/Machine learning tuts/Target/Part2(Deep Learning)/DeepLearning-Part2/train.csv')
  data=df.as_matrix().astype(np.float32)
  np.random.shuffle(data)  
  X=data[:,1:]
  mu=X.mean(axis=0)
  std=X.std(axis=0)
  np.place(std, std == 0, 1)
  X=(X-mu)/std
  Y=data[:,0]
  return X,Y
  

def forward(X,W,b):
  #softmax
  A=X.dot(W)+b
  expA=np.exp(A)
  return expA/expA.sum(axis=1,keepdims=True)

def predict(pY):  
  return np.argmax(pY,axis=1)
  
def errorRate(predictions,target):
  return np.mean(target!=predict(predictions))

def cost(pY,target):
  return -((target*np.log(pY)).sum())

def gradW(y,target,X):
  return X.T.dot(y-target)
  
def gradb(y,t):
  return (y-t).sum(axis=0)
  
def y2indicator(y):
  N=len(y)
  y=y.astype(np.int32)
  ind=np.zeros((N,10))
  for i in range(N):
    ind[i,y[i]]=1
  return ind

  
  

  
  
  