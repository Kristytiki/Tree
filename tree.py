#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Author: Zheqi Wu
 Date : 06/04/2018
 Description: one layer tree for matching pursuit
"""
import numpy as np
import matplotlib.pyplot as plt
import random


def re_tree(X_train, Y_train,lambda_): 
    
    Loss=np.zeros((X_train.shape[0],X_train.shape[1]))
    result=np.zeros((1000,4))
    loss0=0
    loss1=1
    k=0
    train=np.c_[X_train,Y_train]
    while(abs(loss1-loss0)>lambda_):
        for j in range(0,X_train.shape[1]):
            temp=train[np.lexsort((train[:, j], ))]
            X_train_=temp[:,0:2]
            Y_train_=temp[:,2]
            for i in range(0,X_train.shape[0]):
                a1=np.mean(Y_train_[0:(i+1)])
                a2=np.mean(Y_train_[(i+1):])
                Loss[i,j]=sum((Y_train_[0:(i+1)]-a1)**2)+sum((Y_train_[(i+1):]-a2)**2)
        ij_min = np.where(Loss == Loss.min())
        ij_min = tuple([i.item() for i in ij_min])
        #print(ij_min)
        temp=train[np.lexsort((train[:, ij_min[1]], ))]
        X_train=temp[:,0:2]
        Y_train=temp[:,2]
        res=np.r_[Y_train[0:(ij_min[0]+1)]-np.mean(Y_train[0:(ij_min[0]+1)]),Y_train[(ij_min[0]+1):]-np.mean(Y_train[(ij_min[0]+1):])]
        train=np.c_[X_train,res]
        loss0=loss1
        #print(loss0)
        loss1=Loss.min()
        #print(loss1)
        result[k]=np.c_[np.mean(Y_train[0:(ij_min[0]+1)]),np.mean(Y_train[(ij_min[0]+1):]),X_train[ij_min[0]][ij_min[1]],ij_min[1]]
        k=k+1
        
    return result

#print(Y_test)
#tree=re_tree(X_train, Y_train)

def tree_predict(X_train, X_test, Y_train, Y_test,lambda_):
    tree=re_tree(X_train, Y_train,lambda_)
    res=Y_test.copy()
    n=np.where(tree[:,0]==0)[0][0]-1
    f_x=np.zeros((X_test.shape[0],1))
    
    for i in range(0,X_test.shape[0]):
        for j in range(0,n):
            if(X_test[i][int(tree[j,3])]>tree[j,2]):
                res[i]=res[i]-tree[j,1]
                f_x[i]=f_x[i]+tree[j,1]
            else:
                res[i]=res[i]-tree[j,0]
                f_x[i]=f_x[i]+tree[j,0]
            #print(res[i])
    return f_x




