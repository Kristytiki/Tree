#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Author: Zheqi Wu
 Date : 06/04/2018
 Description: nerual network
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import train_test_split
## regression
random.seed(1234)

def Relu(A,deriv='TRUE'):
    n,p=A.shape
    B = A.copy()
    for i in range(0,n):
        for j in range(0,p):
            
            if (B[i,j]<0):
                B[i,j]=0
            else:
                if (deriv=='TRUE'):
                    B[i,j]=1
                    
    return B

def tanh(x):
    return  (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def forward(X_train,W,beta):
    S=np.dot(X_train,W)
    H=Relu(S,deriv='FALSE')
    f_x=np.dot(H,beta)
    
    return S,H,f_x

def backward(X_train,Y_train,f_x,beta,H,S,W,rate_=0.01):
    ## learn beta
    db=np.dot(H.T,(Y_train-f_x))
    ## learn W
    dh=np.dot((Y_train-f_x),beta.T)  # n*d
    dw=np.dot(X_train.T,np.multiply(Relu(S),dh)) # p*d
    
    ##update
    beta=beta+rate_*db
    W=W+rate_*dw
    
    return beta,W

def one_nn(X_train, X_test, Y_train, Y_test,reg='FLASE'):
    
    ## forward
    n,p=X_train.shape
    d=20
    X_train=np.c_[np.ones(n),X_train]
    # initialize 
    #W=np.reshape(np.random.randn((p+1)*d,1),(p+1,d))* 0.01
    W=np.reshape(np.random.uniform(0,1,(p+1)*d),(p+1,d))* 0.01
    beta=np.reshape(np.zeros(d),(d,1))
    W0=W.copy()+1
    err_trains=[]
    k=0
    while(k<500):
        W0=W.copy()
        S,H,f_x=forward(X_train,W,beta)
        if (reg!='TRUE'):
            f_x=tanh(f_x)
        #print(f_x[2:5])
        beta,W=backward(X_train,Y_train,f_x,beta,H,S,W,rate_=0.01)
        err = np.mean((Y_train-f_x)**2)
        print(err)
        err_trains.append(err)  
        k=k+1
    
        ## predict
        X_test1=np.c_[np.ones(X_test.shape[0]),X_test]
        _, _, yhat_test=forward(X_test1,W,beta)
        
    if(reg=='TRUE'):

        err=np.mean((Y_test-yhat_test)**2)
        ## plot
        plt.plot(np.arange(0,30,1), Y_test,color='black')
        plt.plot(np.arange(0,30,1), yhat_test,color='red',linestyle='dashed')
        plt.legend(["Y_test","Y_fitted"], ncol=4, loc='upper center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    else:
        yhat_test1=tanh(yhat_test)+0.001
        f_x=np.sign(yhat_test1)
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
    plt.show()
    '''
    S=np.dot(X_train,W)
    H=Relu(S,deriv='FALSE')
    f_x=np.dot(H,beta)
    
    
    ## backward
    ## learn beta
    db=np.dot(H.T,(Y_train-f_x))
    
    ## learn W
    dh=np.dot((Y_train-f_x),beta.T)  # n*d
    dw=np.dot(X_train.T,np.multiply(Relu(S),dh)) # p*d
    
    ##update
    beta=beta+rate_*db
    W=W+rate_*dw
    
    '''
    
    
    
    return 0





random.seed(1234)
## regression setting
n=150
#p=100
rate_ = 0.01
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
X = np.c_[x1,x2]
Y = np.reshape((x1**2)+(x2**2),(n,1)) + np.random.randn(n,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

one_nn(X_train, X_test, Y_train, Y_test,reg='TRUE')


## testing setting

random.seed(1234)

n=150
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
X = np.c_[x1,x2]
Y = np.sign(np.reshape((x1**2)+(x2**2)-1,(n,1)))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lambda_=0.01

one_nn(X_train, X_test, Y_train, Y_test,reg='FALSE')