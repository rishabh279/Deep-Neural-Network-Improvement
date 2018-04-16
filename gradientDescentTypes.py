# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:59:55 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import getTransformedData,forward,errorRate,cost,gradW,gradb,y2indicator

def main():
  X,Y,_,_=getTransformedData()
  X=X[:,:300]
  
  #normalize X 
  mu=X.mean(axis=0)
  std=X.std(axis=0)
  X=(X-mu)/std

  print("Performing logistic regression")
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)  
  N,D=X.shape
  
  ###########''''''FULL Gradient Descent'''''''''''''''''##############
  
  W=np.random.randn(D,10)/28
  b=np.zeros(10)
  
  costs=[]
  learning_rate=0.0001
  reg=0.01
  t0 = datetime.now()
  for i in range(200):
    pY=forward(Xtrain,W,b)
    
    W-=learning_rate*(gradW(pY,Ytrain_ind,Xtrain)+reg*W)
    b-=learning_rate*(gradb(pY,Ytrain_ind)+reg*b)
    
    pYtest=forward(Xtest,W,b)
    c=cost(pYtest,Ytest_ind)
    costs.append(c)
    if i%10==0:
      e=errorRate(pYtest,Ytest)
      print("i",i,"cost",c,"error",e)
    
  print("Elaspsed Time",datetime.now()-t0)
  
  ###########''''''''''''''Stochastic Gradient Descent'''''''''#############
  
  W=np.random.randn(D,10)   
  b=np.zeros(10)
  cost_stochastic=[]
  learning_rate=0.0001
  reg=0.01
  
  t0=datetime.now()
  for i in range(1):
    tmpX,tmpY=shuffle(Xtrain,Ytrain_ind)
    for n in range(min(N,500)):
      x=tmpX[n,:].reshape(1,D)
      y=tmpY[n,:].reshape(1,10)
      pY=forward(x,W,b)
      W-=learning_rate*(gradW(pY,y,x)+reg*W)
      b-=learning_rate*(gradb(pY,y)+reg*b)
      
      pYtest=forward(Xtest,W,b)
      c=cost(pYtest,Ytest_ind)
      cost_stochastic.append(c)
      e=errorRate(pYtest,Ytest)
      if n%2==0:
        print("Error",e,"Cost",c,"i",n)
  print("Elaspsed Time",datetime.now()-t0)
  
  ####################################Batch Gradient Descent############################    
  W=np.random.randn(D,10)
  cost_batch=[]
  b=np.zeros(10)
  learning_rate=0.0001
  reg=0.01
  batch_sz=500
  n_batch=N//batch_sz
  
  t0=datetime.now()
  for i in range(50):
    tmpX,tmpY=shuffle(Xtrain,Ytrain_ind)
    for j in range(n_batch):
      x=tmpX[j*batch_sz:(j*batch_sz+batch_sz),:]
      y=tmpY[j*batch_sz:(j*batch_sz+batch_sz),:]

      pY=forward(x,W,b)
      
      W-=learning_rate*(gradW(pY,y,x)+reg*W)
      b-=learning_rate*(gradb(pY,y)+reg*b)
      
      pYtest=forward(Xtest,W,b)
      c=cost(pYtest,Ytest_ind)
      cost_batch.append(c)
      error=errorRate(pYtest,Ytest)
      if j%(n_batch)==0:
        print("j ",j,"cost ",c,"error ",error)
  print("Time taken",datetime.now()-t0)
    
if __name__=='__main__':
  main()
  

