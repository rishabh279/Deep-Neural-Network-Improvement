# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:57:28 2018

@author: rishabh
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import getNormalizedData,errorRate,cost,y2indicator

def forward(X,W1,b1,W2,b2):
  #sigmoid
  Z=1/(1+np.exp(-(X.dot(W1)+b1)))
  expA=np.exp(Z.dot(W2)+b2)
  return expA/expA.sum(axis=1,keepdims=True),Z

def derivative_W2(Y,T,Z):
  return Z.T.dot(Y-T)
  
def derivative_b2(Y,T):
  return (Y-T).sum(axis=0)
  
def derivative_W1(Y,T,W2,Z,X):
  dz=(Y-T).dot(W2.T)*Z*(1-Z)
  return X.T.dot(dz)
  
def derivative_b1(Y,T,W2,Z):
  return ((Y-T).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
  
def main():
  
  max_iter=20
  print_period=10
  
  X,Y=getNormalizedData()
  learning_rate=0.00004
  reg=0.01
  
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)

  N,D=X.shape
  batch_sz=500
  n_batches=N//batch_sz
  
  
  M=300
  K=len(set(Y))
  W1=np.random.randn(D,M)/np.sqrt(D)
  b1=np.zeros(M)
  W2=np.random.randn(M,K)/np.sqrt(M)
  b2=np.zeros(K)
  
  W1_0=W1.copy()
  W2_0=W2.copy()
  b1_0=b1.copy()
  b2_0=b2.copy()
  
  
  #Batch Gradient Descent without Momemtum
  
  costs=[]
  for i in range(max_iter):
    for j in range(n_batches):
      batchX=Xtrain[j*batch_sz:(batch_sz+j*batch_sz)]
      batchY=Ytrain_ind[j*batch_sz:(batch_sz+j*batch_sz)]
      pY,Z=forward(batchX,W1,b1,W2,b2)
            
      W2-=learning_rate*(derivative_W2(pY,batchY,Z)+reg*W2)
      b2-=learning_rate*(derivative_b2(pY,batchY)+reg*b2)
      W1-=learning_rate*(derivative_W1(pY,batchY,W2,Z,batchX)+reg*W1)
      b1-=learning_rate*(derivative_b1(pY,batchY,W2,Z)+reg*b1)
      
      if j%print_period==0:
        pYtest,_=forward(Xtest,W1,b1,W2,b2)
        c=cost(pYtest,Ytest_ind)
        costs.append(c)
        e=errorRate(pYtest,Ytest)
        print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, c))
        print("Error rate:", e)
        
  

  #Batch Gradient Descent with Momemtum
  cost_momemtum=[]
  W1=W1_0.copy()
  W2=W2_0.copy()
  b1=b1_0.copy()
  b2=b2_0.copy()
  mu=0.9
  dW2=0
  dW1=0
  db1=0
  db2=0
  for i in range(max_iter):
    for j in range(n_batches):
      batchX,batchY=Xtrain[j*batch_sz:(batch_sz+j*batch_sz)],Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
      pY,Z=forward(batchX,W1,b1,W2,b2)
      
      #gradients
      gW2=derivative_W2(pY,batchY,Z)+reg*W2
      gb2=derivative_b2(pY,batchY)+reg*b2
      gW1=derivative_W1(pY,batchY,W2,Z,batchX)+reg*W1
      gb1=derivative_b1(pY,batchY,W2,Z)+reg*b1
      
      #update velocities
      dW2=mu*dW2+gW2
      db2=mu*db2+gb2
      dW1=mu*dW1+gW1
      db1=mu*db1+gb1
      
      #Update
      W2-=learning_rate*(dW2)
      b2-=learning_rate*(db2)
      W1-=learning_rate*(dW1)
      b1-=learning_rate*(db1)
      
      if j%print_period==0:
        pYtest,_=forward(Xtest,W1,b1,W2,b2)
        c=cost(pYtest,Ytest_ind)
        cost_momemtum.append(c)
        e=errorRate(pYtest,Ytest)
        print("At i=%d iteration cost is %.6f"%(i,c))
        print("Error is ",e)   
        
if __name__=='__main__':
  main()
                  
      
      
  


