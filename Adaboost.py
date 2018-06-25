#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Author: Zheqi Wu
 Date : 06/04/2018
 Description: Adaboost classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random


random.seed(1224)

n=500
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
X = np.c_[x1,x2]
Y = np.sign(np.reshape((x1**2)+(x2**2)-1,(n,1)))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lambda_=0.01


def adaboost(X_train,Y_train):
    ## weak classifier
    ## calculate D_i
    ## minimize wighted error
    ##i
    
    result=np.zeros((1000,3))
    loss0=0
    loss1=1
    k=0
    train=np.c_[X_train,Y_train]
    beta=np.zeros((1000,1))
    f_x=0
    ## initial weight
    D=np.ones((X_train.shape[0],1))/X_train.shape[0]
    while(k<100):
        error=np.zeros((X_train.shape[0],X_train.shape[1]))
        for j in range(0,X_train.shape[1]):
            temp=train[np.lexsort((train[:, j], ))].copy()
            X_train_=temp[:,0:2].copy()
            Y_train_=temp[:,2].copy()
            #print(j)
            for i in range(0,X_train.shape[0]):
                a1=-1
                a2=1
                flag=0
                ## tree weighted loss
                #print(sum(D[0:(i+1)]*np.reshape((1-(Y_train_[0:(i+1)]*a1))*0.5,(i+1,1))))
                error[i,j]=sum(np.multiply(np.reshape(D[0:(i+1)],(i+1,1)),np.reshape((1-(Y_train_[0:(i+1)]*a1))*0.5,(i+1,1))))+sum(np.multiply(np.reshape(D[(i+1):],(X_train_.shape[0]-i-1,1)),np.reshape((1-(Y_train_[(i+1):]*a2))*0.5,(X_train.shape[0]-i-1,1))))
                if(error[i,j]>0.5):
                    error[i,j]=sum(np.multiply(np.reshape(D[0:(i+1)],(i+1,1)),np.reshape((1-(Y_train_[0:(i+1)]*a2))*0.5,(i+1,1))))+sum(np.multiply(np.reshape(D[(i+1):],(X_train_.shape[0]-i-1,1)),np.reshape((1-(Y_train_[(i+1):]*a1))*0.5,(X_train.shape[0]-i-1,1))))
        ij_min = np.c_[np.where(error == error.min())[0][0],np.where(error == error.min())[1][0]]
        
        print(ij_min)
        #print("error.min")
        #print(error.min())
        beta[k]=0.5*np.log((1-error.min())/error.min())
        temp=train[np.lexsort((train[:, ij_min[0,1]], ))].copy()
        X_train=temp[:,0:2].copy()
        Y_train=temp[:,2].copy()
        
        check_error=sum(np.multiply(np.reshape(D[0:(ij_min[0,0]+1)],(ij_min[0,0]+1,1)),np.reshape((1-(Y_train[0:(ij_min[0,0]+1)]*a1))*0.5,(ij_min[0,0]+1,1))))+sum(np.multiply(np.reshape(D[(ij_min[0,0]+1):],(X_train.shape[0]-ij_min[0,0]-1,1)),np.reshape((1-(Y_train[(ij_min[0,0]+1):]*a2))*0.5,(X_train.shape[0]-ij_min[0,0]-1,1))))
        if(check_error<0.5):
            f_x=f_x+beta[k]*np.r_[-np.ones(ij_min[0,0]+1),np.ones(X_train.shape[0]-ij_min[0,0]-1)]
        else:
            f_x=f_x+beta[k]*np.r_[np.ones(ij_min[0,0]+1),-np.ones(X_train.shape[0]-ij_min[0,0]-1)]
        #print("f_x")
        #print(f_x)
        ## ex loss
        loss0=loss1
        #print("beta")
        #print(beta[k])
        loss1=(1-error.min())*np.exp(-beta[k])+error.min()*np.exp(beta[k])
        print(loss1)
        result[k]=np.c_[beta[k],X_train[ij_min[0,0],ij_min[0,1]],ij_min[0,1]]
        k=k+1
        D=np.exp(-Y_train*f_x)
        D=D/sum(D)
        #print("D")
        #print(D[100:105])
        
    return result

def ada_predict(X_train, X_test, Y_train, Y_test):
    tree = adaboost(X_train,Y_train)
    res=Y_test.copy()
    n=np.where(tree[:,0]==0)[0][0]-1
    f_x=np.zeros((X_test.shape[0],1))
    
    for i in range(0,X_test.shape[0]):
        for j in range(0,n):
            if(X_test[i][int(tree[j,2])]>tree[j,1]):
                f_x[i]=f_x[i]+tree[j,0]*1
            else:
                f_x[i]=f_x[i]+tree[j,0]*(-1)
            #print(res[i])
    return f_x
f_x=np.sign(ada_predict(X_train, X_test, Y_train, Y_test))

X1 = np.arange(0,1,0.01)
X2 = np.arange(0,1,0.01)
X1,X2 = np.meshgrid(X1,X2)
Z = np.sign((X1**2)+(X2**2)-1)
X1_test = np.array(list(zip(*X_test))[0])
X2_test = np.array(list(zip(*X_test))[1])
plt.contourf(X1,X2,Z,colors=('r','b'))
plt.scatter(X1_test,X2_test,s=20,c=f_x)
plt.colorbar()
plt.set_title("decision boundary")
plt.set_xlabel('X0')
plt.set_ylabel('X1') 