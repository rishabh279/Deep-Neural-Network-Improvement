# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:50:02 2018

@author: rishabh
"""
import numpy as np

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
  reg =0.01
  
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)
  
  N,D=Xtrain.shape
  batch_sz=500
  n_batches=N//batch_sz
  
  M=300
  K=10
  W1_0=np.random.randn(D,M)/np.sqrt(D)
  b1_0=np.zeros(M)
  W2_0=np.random.randn(M,K)/np.sqrt(M)
  b2_0=np.zeros(K)
  
  W1=W1_0.copy()
  b1=b1_0.copy()
  W2=W2_0.copy()
  b2=b2_0.copy()
  
  #1st moment
  mW1=0
  mb1=0
  mW2=0
  mb2=0
  
  #2nd moment
  vW1=0
  vb1=0
  vW2=0
  vb2=0
  '''
  #hyperparams
  learning_rate=0.001
  beta1=0.9
  beta2=0.999
  eps=1e-8
  
  #Adam Optimization
  costs_adam=[]
  t=1
  for i in range(max_iter):
    for j in range(n_batches):
      batchX=Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
      batchY=Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
      pY,Z=forward(batchX,W1,b1,W2,b2)
      
      #gradients
      gW2=derivative_W2(pY,batchY,Z)+reg*W2
      gb2=derivative_b2(pY,batchY)+reg*b2
      gW1=derivative_W1(pY,batchY,W2,Z,batchX)
      gb1=derivative_b1(pY,batchY,W2,Z)
      
      # new m
      mW1 = beta1 * mW1 + (1 - beta1) * gW1
      mb1 = beta1 * mb1 + (1 - beta1) * gb1
      mW2 = beta1 * mW2 + (1 - beta1) * gW2
      mb2 = beta1 * mb2 + (1 - beta1) * gb2

      # new v
      vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
      vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
      vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
      vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2

      # bias correction
      correction1 = 1 - beta1 ** t
      hat_mW1 = mW1 / correction1
      hat_mb1 = mb1 / correction1
      hat_mW2 = mW2 / correction1
      hat_mb2 = mb2 / correction1

      correction2 = 1 - beta2 ** t
      hat_vW1 = vW1 / correction2
      hat_vb1 = vb1 / correction2
      hat_vW2 = vW2 / correction2
      hat_vb2 = vb2 / correction2
      
      # update t
      t += 1

      # apply updates to the params
      W1 = W1 - learning_rate * hat_mW1 / np.sqrt(hat_vW1 + eps)
      b1 = b1 - learning_rate * hat_mb1 / np.sqrt(hat_vb1 + eps)
      W2 = W2 - learning_rate * hat_mW2 / np.sqrt(hat_vW2 + eps)
      b2 = b2 - learning_rate * hat_mb2 / np.sqrt(hat_vb2 + eps)
      
      if j% print_period==0:
        pYtest,_=forward(Xtest,W1,b1,W2,b2)
        c=cost(pYtest,Ytest_ind)
        costs_adam.append(c)    
        print("At iteration i=%d j=%d cost is %.6f"%(i,j,c))
        e=errorRate(pYtest,Ytest)
        print("Error Rate",e)
          '''
#Rms Prop with momentum
  W1=W1_0.copy()
  b1=b1_0.copy()
  W2=W2_0.copy()
  b2=b2_0.copy()
  
  cost_rms=[]
  learning_rate_0=0.001
  mu=0.9
  decay_rate=0.999
  eps=1e-8

  #rms cache
  cache_W1=1
  cache_b1=1
  cache_W2=1
  cache_b2=1
  
  dW2=0
  dW1=0
  db1=0
  db2=0
  
  for i in range(max_iter): 
    for j in range(n_batches):
      batchX=Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
      batchY=Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
      pY,Z=forward(batchX,W1,b1,W2,b2)
      
      gW2=derivative_W2(pY,batchY,Z)+reg*W2
      cache_W2=decay_rate*cache_W2+(1-decay_rate)*gW2*gW2
      dW2=mu*dW2+(1-mu)*learning_rate_0*gW2/(np.sqrt(cache_W2)+eps)
      W2-=dW2
      
      gb2=derivative_b2(pY,batchY)+reg*b2
      cache_b2=decay_rate*cache_b2+(1-decay_rate)*gb2*gb2
      db2=mu*db2+(1-mu)*learning_rate_0*gb2/(np.sqrt(cache_b2)+eps)
      b2-=db2
      
      gW1=derivative_W1(pY,batchY,W2,Z,batchX)+reg*W1
      cache_W1=decay_rate*cache_W1+(1-decay_rate)*gW1*gW1
      dW1=mu*dW1+(1-mu)*learning_rate_0*gW1/(np.sqrt(cache_W1)+eps)
      W1-=dW1
      
      gb1=derivative_b1(pY,batchY,W2,Z)+reg*b1
      cache_b1=decay_rate*cache_b1+(1-decay_rate)*gb1*gb1
      db1=mu*db1+(1-mu)*learning_rate_0*gb1/(np.sqrt(cache_b1)+eps)
      b1-=db1      
        
      if j%print_period==0:
        pYtest,_=forward(Xtest,W1,b1,W2,b2)
        c=cost(pYtest,Ytest_ind)
        cost_rms.append(c)
        print("At iteration i=%d j=%d cost is %.6f"%(i,j,c))
        e=errorRate(pYtest,Ytest)
        print("Error rate is",e)
        
        
if __name__=='__main__':
  main()
        

