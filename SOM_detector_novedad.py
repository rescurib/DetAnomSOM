#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:05:37 2021

@author: rodolfo
"""

import numpy as np
import pandas as pd
import math
import numba
import matplotlib.pyplot as plt
plt.style.use("seaborn")

from scipy import signal # Filtrado

import time

#--------------- Datos ------------------
#Carga de datos
df = pd.read_csv('./EEG_DATA/mindMonitor_2020-03-06--06-08-22.csv',
                   parse_dates=['TimeStamp'])

#El archivo 'mindMonitor_2020-03-06--06-08-22.csv' es adecuado para pruebas 

#Lista de nombres de columnas
cols = df.columns.values

#Eliminar filas con NaN en el campos 1:21
df.dropna(inplace=True,subset=cols[1:21])
#----------------------------------------

#--------------- SOM --------------------
#@numba.jit(nopython=True)
def qerror(W,x):
    """
    Parameters
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    x : Arreglo numpy
        Objeto a clasificar

    Returns
    -------
    qerror : DOUBLE
             Error de cuantización
    """
    Q = W.shape[0]
    darray = np.zeros(Q) #array de distancias W[i],X[t]
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
    
    for i in range(Q):
            #--- d[i]:(distancia(w,dato))
            darray[i] = math.sqrt(dot(x-W[i],x-W[i])) 
            
    return np.min(darray)

@numba.jit(nopython=True)
def qerrorn(W,X,depth):
    """
    Parameters
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    x : Arreglo numpy
        Objeto a clasificar

    Returns
    -------
    qerror : DOUBLE
             Error de cuantización
    """
    Q = W.shape[0]
    darray = np.zeros(Q) #array de distancias W[i],X[t]
    aarray = np.zeros(X.shape[0]-depth) #array de anomalias
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
    
    for k in range(X.shape[0]-depth): #iteración a lo largo de la serie
        x = X[k:k+depth] # ventana   
        for i in range(Q): # Iteración sobre neuronas
            #--- d[i]:(distancia(w,dato))
            darray[i] = math.sqrt(dot(x-W[i],x-W[i])) 
        
        aarray[k] = np.min(darray)
            
    return aarray
    
@numba.jit(nopython=True)
def entrenar(W,X,depth,EP,n,a = 0.1,sig0 = 1.5):
    """
    Parámetros
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    x : Arreglo numpy de flotantes
        Dato de entrenamiento.
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
    
    for k in range(X.shape[0]-depth): #iteración a lo largo de la serie
        
        x = X[k:k+depth] # ventana   
    
        for i in range(Q): # Iteración sobre neuronas
            #--- d[i]:(distancia(w,dato))
            darray[i] = math.sqrt(dot(x-W[i],x-W[i])) 
        iganadora = np.argmin(darray)
        
        #Actualización de pesos
        for i in range(Q):
            d = math.sqrt((iganadora-i)**2) 
            Hck = np.exp(-d/(2*(sig0*np.exp(-n/EP))**2))
            W[i] += Hck*a*(x-W[i])
    return W
#----------------------------------------

#---------- Programa principal ----------
depth = 20 # Profundidad de ventana
Q     = 100
W     = np.random.rand(Q,depth) # Matriz de pesos
EP    = 10

# Padding y noermalizado de datos
serieTiempo = df[cols[1]].values
serieTiempo = serieTiempo/serieTiempo.max()
#serieTiempo = np.concatenate((np.zeros(depth-1),serieTiempo),axis=0)
#serieTiempo = np.concatenate((serieTiempo,np.zeros(depth-1)),axis=0)

# Entrenamiento
start = time.time()
for n in range(EP):
    W = entrenar(W,serieTiempo,depth,EP,n)
end = time.time()
print("Entrenamiento:",end - start)

# Detección de anomalías
start = time.time()
an_array = qerrorn(W,serieTiempo,depth)
end = time.time()
print("Detección:",end-start)

an_array = np.array(an_array)
    
#----------------------------------------

# #%% Filtrado
# Fs = 1
# Ts = 1/Fs

# wc = 1/30
# sos = signal.butter(10, wc, 'lp', fs=Fs, output='sos') # Second order sections
# afilt = an_array.copy()#[40::]#signal.sosfilt(sos,an_array[40::])
# #plt.plot(afilt,'.--')
# #plt.show()

#%%Gráficas

fig, ax = plt.subplots(2,1,sharex=True)
xx = df[cols[0]].values[int(depth/2):df.shape[0]-int(depth/2)]
xd = df[cols[0]].values



ax[0].plot(xx,an_array,label="$e^{q}(k)$")
ax[0].set_ylabel("Error de cuantización")
ax[0].legend()

ax[1].plot(xd,serieTiempo,label=cols[1])
ax[1].legend()

threshold = 0.6
ax[0].axhline(threshold, color='red',linestyle='--', lw=2, alpha=0.7)
ax[1].fill_between(xx, 0, 1, where=an_array > threshold,
                color='green', alpha=0.5, transform=ax[1].get_xaxis_transform())
plt.show()

#%%
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import PolyCollection
# from matplotlib.colors import colorConverter

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

# xs = np.arange(0, depth)
# verts = []
# zs = np.arange(100)#[0.0, 1.0, 2.0, 3.0]
# for z in zs:
#     ys = W[z]
#     ys[0], ys[-1] = 0, 0
#     verts.append(list(zip(xs, ys)))

# poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
#                                            cc('y')])
# poly.set_alpha(0.7)
# ax.add_collection3d(poly, zs=zs, zdir='y')

# ax.set_xlabel('X')
# ax.set_xlim3d(0, depth)
# ax.set_ylabel('Y')
# ax.set_ylim3d(0, 100)
# ax.set_zlabel('Z')
# ax.set_zlim3d(0, 1)

# plt.show()


#fig = plt.figure(2)
#ax2 = fig.gca(projection='3d')
#for i in range(W.shape[0]):
#    ax2.plot(i*np.ones(depth),np.arange(depth),W[i])

fig, ax2 = plt.subplots(10,1,sharex=True)
for i in range(10):
    ax2[i].plot(W[i])