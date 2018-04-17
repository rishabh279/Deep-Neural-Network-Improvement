# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:27:34 2018

@author: rishabh
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import getNormalizedData,y2indicator

def errorRate(Y,T):
  return np.mean(Y!=T)
  
def main():
  X,Y=getNormalizedData()
  
  max_iter=15
  print_period=10
  
  reg=0.01
  lr=0.00004
  
  Xtrain=X[:-1000]
  Ytrain=Y[:-1000]
  Xtest=X[-1000:]
  Ytest=Y[-1000:]
  Ytrain_ind=y2indicator(Ytrain)
  Ytest_ind=y2indicator(Ytest)
  
  N,D=X.shape
  batch_sz=500
  n_batches=N//batch_sz
  
  M1=300
  M2=100
  K=10
  
  W1_init=np.random.randn(D,M1)/np.sqrt(D)
  b1_init=np.zeros(M1)
  W2_init=np.random.randn(M1,M2)/np.sqrt(M1)
  b2_init=np.zeros(M2)
  W3_init=np.random.randn(M2,K)/np.sqrt(M2)
  b3_init=np.zeros(K)
  
  #define variables and expressions
  X=tf.placeholder(tf.float32,shape=(None,D),name='X')
  T=tf.placeholder(tf.float32,shape=(None,K),name='T')  
  W1=tf.Variable(W1_init.astype(np.float32))
  b1=tf.Variable(b1_init.astype(np.float32))
  W2=tf.Variable(W2_init.astype(np.float32))
  b2=tf.Variable(b2_init.astype(np.float32))
  W3=tf.Variable(W3_init.astype(np.float32))
  b3=tf.Variable(b3_init.astype(np.float32))

  #define model
  Z1=tf.nn.relu(tf.matmul(X,W1)+b1)
  Z2=tf.nn.relu(tf.matmul(Z1,W2)+b2)  
  Yish=tf.matmul(Z2,W3)+b3
  
  cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish,labels=T))
  
  train_op=tf.train.RMSPropOptimizer(lr,decay=0.99,momentum=0.9).minimize(cost)
  
  predict_op=tf.argmax(Yish,1)
  
  costs=[]
  init=tf.global_variables_initializer()
  with tf.Session() as session:
    session.run(init)
    
    for i in range(max_iter):
      for j in range(n_batches):
        batchX=Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
        batchY=Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
        
        session.run(train_op,feed_dict={X:batchX,T:batchY})
        
        if j%print_period==0:
          test_cost=session.run(cost,feed_dict={X:Xtest,T:Ytest_ind})
          prediction=session.run(predict_op,feed_dict={X:Xtest})
          err=errorRate(prediction,Ytest)
          print("At iteration i=%d j=%d cost is %.6f"%(i,j,test_cost))
          print("Error Rate is",err)
          costs.append(test_cost)
          
  plt.plot(costs)
  # are we overfitting by adding that extra layer?
  # how would you add regularization to this model?
if __name__=='__main__':
  main()
          