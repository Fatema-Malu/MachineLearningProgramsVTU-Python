# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)


y = np.array(([92], [86], [89]), dtype=float)
#print("array max is  : ",np.amax(X,axis=0))
X = X/np.amax(X,axis=0) # maximum of X array longitudinally
print(X)
y = y/100 
print("y :",y)
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable  initialization
epoch=7000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization

wh=np.array([[0.7872,0.4873,0.4041],[0.9057,0.8723,0.9704]])
bh=np.array([[0.0751,0.3708,0.9870]])
wout=np.array([[0.6119],[0.0234],[0.9764]])
bout=np.array([[0.8051]])

#draws a random range of numbers uniformly of dimx*y
for i in range(epoch):

#Forward Propogation
    hinp1=np.dot(X,wh)
    #print("hinp1=   ",hinp1)
    hinp=hinp1 + bh
    #print("hinp=   ",hinp)
    hlayer_act = sigmoid(hinp)
    #print("hlayer act :",hlayer_act,"\n","hlayer_act.T :",hlayer_act.T)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+ bout
    output = sigmoid(outinp)
    #print("output:   ",output)
#Backpropagation
    EO = y-output
    #print("EO=  ",EO)
    outgrad = derivatives_sigmoid(output)
    #print("outgrad= ",outgrad)
    d_output = EO* outgrad    #delta_k
    #print("d_output=  ",d_output)
    EH = d_output.dot(wout.T) #transpose
    #print("wout.T=  ",wout.T)
    #print("EH=  ",EH)  
    hiddengrad = derivatives_sigmoid(hlayer_act)
    #print("hiddengrad=  ",hiddengrad)
    #how much hidden layer wts contributed to error
    d_hiddenlayer = EH * hiddengrad  #delta_h
    #print("d_hiddenlayer=  ",d_hiddenlayer)
    wout += hlayer_act.T.dot(d_output) *lr
    #print("wout after delta:   ",wout)
    # dotproduct of nextlayererror and currentlayerop
    wh += X.T.dot(d_hiddenlayer) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)


