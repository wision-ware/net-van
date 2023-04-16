# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:55:38 2021

@author: vavri
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
import os


class Network(object):
    

#activation functions
    
    def ReLU(x, d=False):
            
        return np.where(x<0,0.,(x if d==False else 1)) 
        
        
    def Sigmoid(x, d=False):

        return (1/(1+np.exp(-x))) if d == False\
        else ((1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))))
        
    
    def __init__(self, par_filename, N):
 
    #processing user input          
        
        bias_load = []
        weight_load = []
        
        for l in range(len(N)):
            
            weight_load.append(np.load(f'{str(par_filename)}_w{l}.npy'))
            bias_load.append(np.load(f'{str(par_filename)}_b{l}.npy'))
        
        params = [weight_load,bias_load]
        self.N = N
        
        weight_like = [0]*len(self.N)
        bias_like = [0]*len(self.N)
        for l in range(1,len(self.N)):    
            weight_like[l] = np.zeros((self.N[l-1],self.N[l]))  
            bias_like[l] = np.zeros((self.N[l]))
        self.weight_like = weight_like
        self.bias_like = bias_like
        self.weights = self.weight_like[:]
        self.bias = self.bias_like[:]
            
        for l,neurons in enumerate(self.N):
                
            self.weights[l] = params[0][l]
            self.bias[l] = params[1][l]

#output method
    
    def get_output(self, inp_, layer=False, label=None):
            
        if layer == True:            
            all_out = []

    #first layer output
                
        p_output = inp_[:]
        
        if layer == True:                    
            all_out.append(p_output[:])
 
    #rest of the layers propagating       
 
        if (layer > 1) or (len(self.N) > 1):
 
            for l in range(1,(len(self.N) if type(layer) == bool else layer)):                
                        
                activation1 = np.sum(np.full(
                    (self.N[l],self.N[l-1]),
                    p_output
                    ).T\
                    *self.weights[l],axis=0)
                p_output = (Network.ReLU(activation1 + self.bias[l]))\
                    if l < (len(self.N)-1) else (Network.Sigmoid(activation1 + self.bias[l]))                      
                        
                if layer == True:
                            
                    all_out.append(p_output[:])
    
    #computing cost
        
        if label is not None:
        
            cost = 0                       
            cost = np.sum((label[:] - (
                p_output[:] if layer==False or type(layer)==int else all_out[-1]))**2)                       
            self.cost = cost                     
            
        return p_output if layer==False or type(layer)==int else all_out  
                        

