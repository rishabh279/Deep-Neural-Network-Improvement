# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:02:02 2018

@author: rishabh
"""
import numpy as np
import matplotlib.pyplot as plt

from util import getNormalizedData,errorRate,cost,y2indicator

def forward(X,W1,b1,W2,b2):
  Z=1/(1+np.exp(-(X.dot(W1)+b1)))
  
  A=Z.dot(W2)+b2
  expA=np.exp(A)
  return expA/(expA.sum(axis=1,keepdims=True)),Z
  
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
  X,Y=getNormalizedData()
  
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)
  
  
  learning_rate=0.00004
  max_iter=20
  print_period=10
  reg = 0.01
  
  N,D=Xtrain.shape
  batch_sz=500
  n_batches=N//batch_sz
  
  M=300
  K=len(set(Y))
  
  W1=np.random.randn(D,M)/np.sqrt(D)
  b1=np.zeros(M)
  W2=np.random.randn(M,K)/np.sqrt(M) 
  b2=np.zeros(K)
  
  
  #Batch Gradient Descent
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
        print("At iteration i=%d j=%d cost is %.6f"%(i,j,c))
        print("Error is",e)
       
  #RMS Prop  
  cost_rms=[]
  W1=np.random.randn(D,M)/np.sqrt(D)
  b1=np.zeros(M)
  W2=np.random.randn(M,K)/np.sqrt(M)
  b2=np.zeros(K)
  cache_W1=1
  cache_W2=1
  cache_b1=1
  cache_b2=1
  
  learning_rate_0=0.001
  decay_rate=0.999
  eps=1e-10
  for i in range(max_iter):
    for j in range(n_batches):
      batchX=Xtrain[j*batch_sz:(batch_sz+(j*batch_sz))]
      batchY=Ytrain_ind[j*batch_sz:(batch_sz+(j*batch_sz))]
      
      pY,Z=forward(batchX,W1,b1,W2,b2)
      
      #gradients
      gW2=derivative_W2(pY,batchY,Z)+reg*W2
      cache_W2=decay_rate*cache_W2+(1-decay_rate)*gW2*gW2
      W2-=learning_rate_0*gW2/(np.sqrt(cache_W2)+eps)
      
      gb2=derivative_b2(pY,batchY)+reg*b2
      cache_b2=decay_rate*cache_b2+(1-decay_rate)*gb2*gb2
      b2-=learning_rate_0*gb2/(np.sqrt(cache_b2)+eps)
      
      gW1=derivative_W1(pY,batchY,W2,Z,batchX)+reg*W1
      cache_W1=decay_rate*cache_W1+(1-decay_rate)*gW1*gW1
      W1-=learning_rate_0*gW1/(np.sqrt(cache_W1)+eps)
      
      gb1=derivative_b1(pY,batchY,W2,Z)+reg*b1
      cache_b1=decay_rate*cache_b1+(1-decay_rate)*gb1*gb1
      b1-=learning_rate_0*gb1/(np.sqrt(cache_b1)+eps)
      
      if j%print_period==0:
        pYtest,_=forward(Xtest,W1,b1,W2,b2)
        c=cost(pYtest,Ytest_ind)
        cost_rms.append(c)
        e=errorRate(pYtest,Ytest)
        print("At iteration i=%d j=%d costs is %.6f"%(i,j,c))
        print("Error is",e)
        
  plt.plot(costs, label='const')
  plt.plot(cost_rms, label='rms')
  plt.legend()
  plt.show()

if __name__=='__main__':
  main()
        
      
      
