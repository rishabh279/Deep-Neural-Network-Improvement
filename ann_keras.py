# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:02:58 2018

@author: rishabh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix


dataset=pd.read_csv('')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

labelencoder_X1=LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.transform(Xtest)

classifier=Sequential()

#first hidden layer
#6 becoz of (no of independent variable +dependent variables)/2
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

#adding second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#adding output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(Xtrain,Ytrain,batch_size=10,epochs=100)

#Making predictions
Ypred=classifier.predict(Xtest)
Ypred=(Ypred>0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction=classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction=new_prediction>0.5

cm=confusion_matrix(Ytest,Ypred)

#Evaluating Improving and Tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initalizer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_intializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=Xtrain,y=Ytrain,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()
        
#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initalizer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initaliazer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                             scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(Xtrain,Ytrain)
bestparameters=grid_search.best_params_
best_accuracy=grid_search.best_score_