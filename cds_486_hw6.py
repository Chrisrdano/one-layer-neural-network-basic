# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:55:27 2019

@author: Chris
"""

"""
CDS 486: Intro to Computational Learning
Homework 6
Back-propagation
"""
import numpy as np
from math import *

# desired output
y = 0.85              # Y is the target output for the neural network
 
eta = 0.10            # eta is the learning rate. 
                      # In this case, a value was specified
# inputs and weights
inputs = np.matrix([[1.00],[0.95],[0.57],[0.01],[0.25]])

#weights1 = np.matrix([[.15],[-.45],[.27],[.93],[-.75]])
#weights2 = np.matrix([[.37],[.32],[-.61],[-.84],[.48]])
#weights3 = np.matrix([[.01],[-.91],[.94]])

weights = np.matrix([[.15],[-.45],[.27],[.93],[-.75],
                    [.37],[.32],[-.61],[-.84],[.48],
                    [.01],[-.91],[.94]])
#############
# First Forward pass
z1 = np.dot(np.transpose(weights[0:5,]), inputs) 
z2 = np.dot(np.transpose(weights[5:10,]), inputs)

f1 = exp(z1[0,0])/(1+exp(z1[0,0]))
f2 = exp(z2[0,0])/(1+exp(z2[0,0]))

f = np.matrix([[1],[f1],[f2]])

t = np.dot(np.transpose(weights[10:13]), f)
print(t)

# Backpropagation (calculate new weights)
for i in range(10,13):
    weights[i,] = weights[i,] + eta*(t - y)*(f[i-10,])
for i in range(0,5):
    weights[i,] = weights[i,] + eta*(t - y)*(y*(1 - y))*inputs[i,]
for i in range(5,10):
    weights[i,] = weights[i,] + eta*(t - y)*(y*(1 - y))*inputs[i-5,]

# Second forward pass 
z1 = np.dot(np.transpose(weights[0:5,]), inputs) 
z2 = np.dot(np.transpose(weights[5:10,]), inputs)

f1 = exp(z1[0,0])/(1+exp(z1[0,0]))
f2 = exp(z2[0,0])/(1+exp(z2[0,0]))

f = np.matrix([[1],[f1],[f2]])

t = np.dot(np.transpose(weights[10:13]), f)
print(t)

