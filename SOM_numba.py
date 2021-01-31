#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:25:37 2021

@author: rodolfo
"""

import numpy as np
import math
import numba
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import time

def predict(W,x):
    d = []
    for i in range(M*N):
        dt = (i,np.linalg.norm(x-W[i])) 
        d.append(dt)
    return min(d,key= lambda x: x[1])[0]

@numba.jit(nopython=True)
def entrenar(W,X,Nep,n,a = 0.1,sig0 = 1.5):
    """
    Parámetros
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    X : Arreglo numpy de flotantes
        Conjunto de datos de entrenamiento.
    n : Entero escalar
        Número de época

    Returns
    -------
    W : Arreglo lineal de flotantes
        Matriz de pesos entrenada.
    """
    Q = W.shape[0]
    darray = np.zeros(Q) #array de distancias W[i],X[t]
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
        
    for t in range(X.shape[0]): #iteración sobre datos de entrenamiento
        for i in range(Q):
            #--- d[i]:(distancia(w,dato))
            darray[i] = math.sqrt(dot(X[t]-W[i],X[t]-W[i])) 
        iganadora = np.argmin(darray)
        
        #Actualización de pesos
        for i in range(Q):
            d = math.sqrt((iganadora-i)**2) 
            Hck = np.exp(-d/(2*(sig0*np.exp(-n/EP))**2))
            W[i] += Hck*a*(data[t]-W[i])
    return W
    

# Cargar dataset
#wines = load_wine() 
#data = wines.data

data, y = make_blobs(n_samples=178, centers=3, n_features=2,
                   random_state=0,cluster_std=0.2)


# Preprocesar datos

data = data / data.max(axis=0)

#%%

#--- SOM ---
#  Dimensiones
M = 1
N = 5

#  Matriz de pesos
W = np.random.rand(M*N,data.shape[1])

#  Hiperparámetros
a = 0.1
sig0 = 1.5
EP = 500

start = time.time()
#  Entrenamiento
for n in range(EP): #iteración sobre épocas
    W = entrenar(W,data,EP,n)
end = time.time()
print("Entrenamiento:",end - start)
#---------
#%%
A = np.zeros(M*N)
for i in range(M*N):
    A[i] = np.linalg.norm(W[i])

#A = A.reshape([M,N])


plt.scatter(data[:,0],data[:,1])
plt.scatter(W[:,0], W[:,1])