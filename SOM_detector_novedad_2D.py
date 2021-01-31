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

@numba.jit(nopython=True)
def qerrorn(W,X):
    """
    Parameters
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    X : Arreglo numpy
        Objeto a clasificar

    Returns
    -------
    qerror : DOUBLE
             Error de cuantización
    """
    M,N,depth = W.shape
    darray = np.zeros(M*N) #array de distancias W[i],X[t]
    aarray = np.zeros(X.shape[0]-depth) #array de anomalias
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
    
    for k in range(X.shape[0]-depth): #iteración a lo largo de la serie
        x = X[k:k+depth] # ventana   
        for i in range(M): # Iteración sobre neuronas
            for j in range(N):
                #--- d[i]:(distancia(w,dato))
                darray[i*N+j] = math.sqrt(dot(x-W[i,j],x-W[i,j])) 
        
        aarray[k] = np.min(darray)
            
    return aarray
    
@numba.jit(nopython=True)
def entrenar(W,X,EP,n,a0 = 0.1,sig0 = 1.5):
    """
    Parámetros
    ----------
    W : Arreglo numpy de dobles
        Matriz de pesos.
    X : Arreglo numpy de dobles
        Serie de tiempo de entrenamiento.
    n : Entero escalar
        Número de época

    Returns
    -------
    W : Arreglo lineal de flotantes
        Matriz de pesos entrenada.
    """
    M,N,depth = W.shape
    Q = M*N 
    K = X.shape[0]-depth
    darray = np.zeros(Q) #array de distancias W[i],X[t]
    lmd = 0.2
    aa = np.zeros(Q) #array de activaciones pasadas
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
    
    for k in range(K): #iteración a lo largo de la serie
        
        x = X[k:k+depth] # ventana   
    
        for i in range(M): # Iteración sobre neuronas
            for j in range(N):
                #--- d[i]:(distancia(w,dato))
                e = math.sqrt(dot(x-W[i,j],x-W[i,j]))
                atm = lmd*aa[i*N+j] - 1/2*e
                darray[i*N+j] = atm
                aa[i*N+j] = atm
        iganadora = np.argmax(darray)
        
        #Actualización de pesos
        for i in range(M):
            for j in range(N):
                rw = np.array([iganadora%M,iganadora/N])
                ri = np.array([i%M,i/N])
                d = math.sqrt(dot(rw-ri,rw-ri)) 
                Hck = np.exp(-d/(2*(sig0*np.exp(-k/K))**2))
                a = a0*(1-k/K)
                W[i,j] += Hck*a*(x-W[i,j])
    return W
#----------------------------------------

#---------- Programa principal ----------
depth = 20 # Profundidad de ventana
M     = 8
N     = 8
W     = np.random.rand(M,N,depth) # Matriz de pesos
EP    = 10

# Normalizado de datos
serieTiempo = df[cols[1]].values
serieTiempo = serieTiempo/serieTiempo.max()


# Entrenamiento
start = time.time()
for n in range(EP):
    W = entrenar(W,serieTiempo,EP,n,a0=1)
end = time.time()
print("Entrenamiento:",end - start)

#%%anomalia artificial
serieTiempo = df[cols[1]].values
serieTiempo = serieTiempo/serieTiempo.max()
anN = 40
serieTiempo[4326:4326+anN] = np.ones(40)*serieTiempo[4326:4326+40].mean()
#anart = np.linspace(-0.2,0.2,15)
tart = np.arange(0,40)
anart = 0.25*np.sin(2*np.pi*0.05*tart)
#anart = np.zeros(20)
#anart[0:10] = np.linspace(-0.2,0.2,10)
#anart[10:20] = np.linspace(0.2,-0.2,10)
serieTiempo[4326:4326+40] += anart

# Detección de anomalías
start = time.time()
an_array = qerrorn(W,serieTiempo)
end = time.time()
print("Detección:",end-start)

an_array = np.array(an_array)
    

#%%Gráficas

fig, ax = plt.subplots(2,1,sharex=True)
xx = df[cols[0]].values[int(depth/2):df.shape[0]-int(depth/2)]
xd = df[cols[0]].values


ax[0].plot(xx,an_array,label="$e^{q}_{i*}(k)$")
ax[0].set_ylabel("Error de cuantización")
ax[0].legend()

ax[1].plot(xd,serieTiempo,label=cols[1])
ax[1].legend()

b = 0.1
threshold = np.percentile(an_array,100*(1-b/2))

ax[0].axhline(threshold, color='red',linestyle='--', lw=2, alpha=0.7)
ax[1].fill_between(xx, 0, 1, where=an_array > threshold,
                color='green', alpha=0.5, transform=ax[1].get_xaxis_transform())
plt.show()

#%%

fig, ax2 = plt.subplots(M,N,sharex=True)
fig.suptitle('Onda Delta (canal TP9)', fontsize=16)
for i in range(M):
    for j in range(N):
        ax2[i,j].plot(W[i,j])
        ax2[i,j].xaxis.set_ticklabels([])
        ax2[i,j].yaxis.set_ticklabels([])
        ax2[i,j].set_ylim(0,0.7)
        
#%%
#@numba.jit(nopython=True)
def find_match(W,X,neuron_id):
    """
    Parameters
    ----------
    W : Arreglo lineal numpy de flotantes
        Matriz de pesos.
    X : Arreglo numpy
        Objeto a clasificar
    neuron_id : Arreglo numpy [2,1]
                Posición de la neurona de patrón.
    Returns
    -------
    qerror : DOUBLE
             Error de cuantización
    """
    M,N,depth = W.shape
    match_array = np.zeros(X.shape[0]-depth) #array de emparejamientos
    i,j = neuron_id
    
    def dot(X0,X1): #Producto punto (np.dot no soportado por numba)
        s = 0
        for i in range(X0.shape[0]):
            s += X0[i]*X1[i]
        return s
    
    for k in range(X.shape[0]-depth): #iteración a lo largo de la serie
        x = X[k:k+depth] # ventana   
        match_array[k] = math.sqrt(dot(x-W[i,j],x-W[i,j])) 
        
    return -match_array

# matches = find_match(W,serieTiempo,np.array([0,0]))

# fig, ax = plt.subplots()
# xx = df[cols[0]].values[int(depth/2):df.shape[0]-int(depth/2)]
# xd = df[cols[0]].values



# ax.plot(xd,serieTiempo,label=cols[1])
# ax.legend()

# ax.fill_between(xx, 0, 1, where = matches > -0.7,
#                 color='green', alpha=0.5, transform=ax.get_xaxis_transform())
# plt.show()