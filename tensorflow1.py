# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:35:44 2018

@author: rishabh
"""

import numpy as np
import tensorflow as tf

A=tf.placeholder(tf.float32,shape=(5,5),name='A')

V=tf.placeholder(tf.float32)

w=tf.matmul(A,V)

with tf.Session() as session:
  output=session.run(w,feed_dict={A:np.random.randn(5,5),V:np.random.randn(5,1)})
  
  print(output,type(output))
  
shape=(2,2)
x=tf.Variable(tf.random_normal(shape))
t=tf.Variable(0)

init=tf.global_variables_initializer()

with tf.Session() as session:
  out=session.run(init)
  print(out)
  
  print(x.eval())
  print(t.eval())
  
u=tf.Variable(20.0)
cost=u*u+u+1.0

train_op=tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init=tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  
  for i in range(12):
    session.run(train_op)
    print("i=%d,cost=%.3f,u=%.3f"%(i,cost.eval(),u.eval()))
    
