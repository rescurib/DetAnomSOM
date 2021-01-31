#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:48:11 2021

@author: rodolfo
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def sig(n,EP,sig0):
    return sig0*np.exp(-n/EP)

def H(r_winner,r_i,n,EP,sig0=1.5):
    d = np.linalg.norm(r_winner-r_i)
    return np.exp(-d/(2*sig(n,EP,sig0)**2))

def predict(W,x):
    d = []
    for i in range(M*N):
        dt = (i,np.linalg.norm(x-W[i])) 
        d.append(dt)
    return min(d,key= lambda x: x[1])[0]
    

# Cargar dataset
#wines = load_wine() 
#data = wines.data

data, y = make_blobs(n_samples=178, centers=3, n_features=2,
                   random_state=0,cluster_std=0.2)


# Preprocesar datos

data = data / data.max(axis=0)


#--- SOM ---
#  Dimensiones
M = 1
N = 5

#  Matriz de pesos
W = np.random.rand(M*N,data.shape[1])

#  Hiperparámetros
a = 0.1
sig0 = 1.5
EP = 300

#  Entrenamiento
for n in range(EP): #iteración sobre épocas
    for t in range(data.shape[0]): #iteración sobre datos de entrenamiento
        d = []
        for i in range(M*N):
            #--- d[i]:(índice de neurona, distancia(w,dato))
            dt = (i,np.linalg.norm(data[t]-W[i])) 
            d.append(dt)
        iganadora = min(d,key= lambda x: x[1])[0]
        
        #Actualización de pesos
        for i in range(M*N):
            Hck = H(iganadora,i,n,EP,sig0)
            W[i] += Hck*a*(data[t]-W[i])
#---------
#%%
A = np.zeros(M*N)
for i in range(M*N):
    A[i] = np.linalg.norm(W[i])

#A = A.reshape([M,N])


plt.scatter(data[:,0],data[:,1])
plt.scatter(W[:,0], W[:,1])