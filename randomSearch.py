# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:46:26 2018

@author: rishabh
"""

from theano_ann import ANN
from util import get_spiral
from sklearn.utils import shuffle
import numpy as np

def random_search():
  X,Y=get_spiral()
  X,Y=shuffle(X,Y)
  Ntrain=int(0.7*len(X))
  Xtrain,Ytrain=X[:Ntrain],Y[:Ntrain]
  Xtest,Ytest=X[Ntrain:],Y[Ntrain:]

  M=20
  nHidden=2
  log_lr=-4
  log_l2=-2
  max_tries=30
  
  #loop through all possible hyperparameter settings
  best_validation_rate=0
  best_lr=None
  best_l2=None
  
  for _ in range(max_tries):
    model=ANN([M]*nHidden)
    model.fit(Xtrain,Ytrain,learning_rate=10**log_lr,reg=10**log_l2,
              mu=0.99,epochs=3000,show_fig=False)
    validation_accuracy=model.score(Xtest,Ytest)
    train_accuracy=model.score(Xtrain,Ytrain)
    print(
      "validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" %
        (validation_accuracy, train_accuracy, [M]*nHidden, log_lr, log_l2)
    )
    if validation_accuracy>best_validation_rate:
      best_validation_rate=validation_accuracy
      best_M=M
      best_nHidden=nHidden
      best_lr=log_lr
      best_l2=log_l2
      
      #select new hyperparams
      nHidden=best_nHidden+np.random.randint(-1,2)
      nHidden = max(1, nHidden)
      M=best_M+np.random.randint(-1,2)
      M=max(10,M)
      log_lr=best_lr+np.random.randint(-1,2)
      log_l2=best_l2+np.random.randint(-1,2)
      
  print("Best validation accuracy",best_validation_rate)
  print("Best_M",best_M)
  print("Best_nHidden",best_nHidden)
  print("learning_rate",best_lr)
  print("l2",best_l2)
  
if __name__=='__main__':
  random_search()
     
  
  