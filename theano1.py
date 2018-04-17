# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:53:58 2018

@author: rishabh
"""

import theano.tensor as T

#different types of variables
c=T.scalar('c')
v=T.vector('v')
A=T.matrix('A')

#matrix multiplication
w=A.dot(v)


import numpy as np

A_val=np.array([[1,2],[3,4]])
v_val=np.array([5,6])


import theano

matrix_times_vector=theano.function(inputs=[A,v],outputs=w)
w_val=matrix_times_vector(A_val,v_val)
print(w_val)

#since normal variables are not updtable we create shared variable

x=theano.shared(20.0,'x')
# a cost function that has a minimum value
cost=x*x+x+1
'''
in theano, we don't have to compute gradients yourself!
first parameter is the parameter we want to take derivative and second parameter is the 
variable with which we want to take gradent with respect to eg w1,w2.Second parameter can take
multiple variables
'''
x_update=x-0.3*T.grad(cost,x)

train=theano.function(inputs=[],outputs=cost,updates=[(x,x_update)])

for i in range(25):
  cost_val=train()
  print(cost_val)
  
print(x.get_value())