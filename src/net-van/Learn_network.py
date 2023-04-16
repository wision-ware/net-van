# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:49:56 2023

@author: vavri
"""

import numpy as np
import time

class Learn_network(object):

    def ReLU(x, d=False):
            
        return np.where(x<0,0.,(x if d==False else 1)) 
        
        
    def Sigmoid(x, d=False):

        return (1/(1+np.exp(-x))) if d == False\
        else ((1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))))
    
    
    def __init__(self, N):

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
            
    
    def get_output(self, inp_, layer=False, label=None):
        
        if layer == True:
            
            all_out_act = []
            all_out = []

    #first layer output
                
        p_output = inp_[:]
        
        if layer == True:
                    
            all_out_act.append(p_output[:])
            all_out.append(p_output[:])
            
 
    #rest of the layers propagating       
 
        if (layer > 1) or (len(self.N) > 1):
 
            for l in range(1,(len(self.N) if type(layer) == bool else layer)):                
                        
                activation1 = np.sum(np.full(
                    (self.N[l],self.N[l-1]),
                    p_output
                    ).T\
                        *self.weights[l],axis=0)
                p_output = (Learn_network.ReLU(activation1 + self.bias[l]))\
                    if l < (len(self.N)-1) else (Learn_network.Sigmoid(activation1 + self.bias[l]))                      
                        
                if layer == True:
                            
                    all_out_act.append(activation1[:])
                    all_out.append(p_output[:])
                    
    
    #computing cost
        
        if label is not None:
        
            cost = 0                       
            cost = np.sum((np.copy(label) - (
                np.copy(p_output) if layer==False or type(layer)==int else all_out[-1]))**2)                       
            self.cost = cost                     
            
        return np.copy(p_output) if layer==False or type(layer)==int else [np.copy(all_out_act),np.copy(all_out)]

#backpropagation
    
    def backpropagate(self, inp, des_out):
                     
        gradient = self.weight_like[:]
        partial_bias = self.bias_like[:]
        output = self.get_output(inp,layer=True,label=des_out)
        
    #output layer       
        
        deltas = Learn_network.Sigmoid(output[0][-1],True)\
        *(output[1][-1]-des_out[:])
        partial_bias_0 = deltas[:]
            
        grad_0 = np.full((self.N[-2],self.N[-1]),deltas)\
        *np.full((self.N[-1],self.N[-2]),output[1][-2]).T
                
        gradient[-1] = grad_0[:]
        partial_bias[-1] = partial_bias_0[:]
        deltas_old = deltas[:]
        
    #hidden layers                     
    
        for l in range(2,len(self.N)):
                    
            sumation = np.full((self.N[-(l-1)],self.N[-l]),Learn_network.ReLU(output[0][-l][:],True)).T\
            *np.full((self.N[-l],self.N[-(l-1)]),deltas_old)*self.weights[-(l-1)]
            deltas_new = np.sum(sumation,axis=1)[:]
            
            gradient[-l] = np.full((self.N[-(l+1)],self.N[-l]),deltas_new)\
            *np.full((self.N[-l],self.N[-(l+1)]),output[1][-(l+1)]).T
            partial_bias[-l] = deltas_new[:]
            deltas_old = deltas_new[:]
            
        return [gradient[:],partial_bias[:]]


#learning algorithm with optional learning rate, cost treshold and GD methods

    def learn(
            self, 
            inp,
            des_out,
            par_filename, 
            treshold, 
            time_limit=np.infty, 
            GD='mini_b', 
            batch_size=50, 
            eta=0.005, 
            live_monitor=False,
            as_text=False,
            fixed_iter=0,
            dia_data=False,
            save_params=True
            ):
                    
        d_index = 0
        gamma = 0.9
        eta_r = 0.001
        self.get_output(inp[0,:],False,label=des_out[0,:])
        avg_cost = 999
        
        if dia_data:
            avg_cost_tracking = []
            avg_eta_tracking = []
        
        if GD == 'mini_b':           
            R_w = self.weight_like[:]
            R_b = self.bias_like[:]
       
        dif_w = self.weight_like[:]
        dif_b = self.bias_like[:]

    #parameter init

        for l in range(1,len(self.N)):                                                
            self.weights[l] = np.random.normal(
            0,2/np.sqrt(self.N[l] + self.N[l-1]),(self.N[l-1],self.N[l]))\
            if l < (self.N[-1]) else np.random.normal(
            0,np.sqrt(2/(self.N[l] + self.N[l-1])),(self.N[l-1],self.N[l])
            )
    
        t_0 = time.process_time()
        elapsed_learning_time = 0
            
    #main training loop        
      
        while (d_index < fixed_iter) if fixed_iter != 0\
            else ((avg_cost > treshold) & (elapsed_learning_time < time_limit)):
            
            if GD == 'stochastic':                
                d_indices = np.arange(len(inp))
                ind = np.random.choice(d_indices)
                partials = self.backpropagate(inp[ind,:],des_out[ind,:])
                avg_cost = self.cost/self.N[-1]
                
            if GD == 'batch':
                
                s_partials = [self.weight_like[:],self.bias_like[:]]
                iter_cost_sum = 0
                
                for i in range(len(inp)):
                    
                    backprop_out = self.backpropagate(inp[i,:],des_out[i,:])
                    iter_cost_sum += self.cost/self.N[-1]               
                    
                    for l in range(1,len(self.N)):                        
                        s_partials[0][l] = s_partials[0][l] + backprop_out[0][l]
                        s_partials[1][l] = s_partials[1][l] + backprop_out[1][l]
                    
                avg_cost = iter_cost_sum/len(inp)
                partials = [
                    [s_layer/len(inp) for s_layer in s_partials[0]],
                    [s_layer/len(inp) for s_layer in s_partials[1]]
                    ] 
                
            if GD == 'mini_b':
                
                s_partials = [self.weight_like[:],self.bias_like[:]]
                d_indices = np.arange(len(inp))
                iter_cost_sum = 0
                
                for i in range(batch_size):
                    
                    ind = np.random.choice(d_indices)
                    backprop_out = self.backpropagate(inp[ind,:],des_out[ind,:])
                    iter_cost_sum += self.cost/self.N[-1]
                    
                    for l in range(1,len(self.N)):                      
                        s_partials[0][l] = s_partials[0][l] + backprop_out[0][l]
                        s_partials[1][l] = s_partials[1][l] + backprop_out[1][l]
                        
                avg_cost = iter_cost_sum/batch_size                      
                partials = [
                    [s_layer/batch_size for s_layer in s_partials[0]],
                    [s_layer/batch_size for s_layer in s_partials[1]]
                    ]    
            
            if dia_data: avg_eta_tracking_ = []
            
            for l in range(1,len(self.N)):
                
                if GD == 'mini_b':                       
                    R_w[l] = (1-gamma)*(partials[0][l])**2\
                        + (gamma*R_w[l] if d_index > 0 else 0)
                        
                dif_w[l] = ((eta_r/(np.sqrt(R_w[l]) + 0.001))*partials[0][l])\
                if GD == 'mini_b' else (eta*partials[0][l] + gamma*dif_w[l])
                self.weights[l] = self.weights[l][:] - dif_w[l][:]
                        
                if GD == 'mini_b':                 
                    R_b[l] = (1-gamma)*(partials[1][l])**2\
                        + (gamma*R_b[l] if d_index > 0 else 0)
                    
                dif_b[l] = ((eta_r/(np.sqrt(R_b[l]) + 0.001))*partials[1][l])\
                if GD == 'mini_b' else (eta*partials[1][l] + gamma*dif_b[l])
                self.bias[l] = self.bias[l][:] - dif_b[l][:]
                
                if dia_data:
                    
                    if GD == 'mini_b':  
                        avg_eta = eta_r/(np.sqrt(R_w[l]) + 0.00001)              
                    else:
                        avg_eta = (eta*partials[0][l] + gamma*dif_w[l])/(partials[0][l]+0.0001)
                    
                    avg_eta = np.average(avg_eta)
                    avg_eta_tracking_.append(avg_eta)

                        
            d_index += 1
            elapsed_learning_time = time.process_time() - t_0
            
            if live_monitor:
                print(avg_cost)
            
            if dia_data:
                avg_cost_tracking.append(avg_cost)
                avg_eta_tracking_ = np.average(np.array(avg_eta_tracking_))
                avg_eta_tracking.append(avg_eta_tracking_)
            
        # os.mkdir(par_filename)
        # os.mkdir(f'{par_filename}')
        
        if save_params:
            for l in range(len(self.N)):        
                np.save(par_filename + '_w' + str(l), self.weights[l],allow_pickle=True)
                np.save(par_filename + '_b' + str(l), self.bias[l],allow_pickle=True)
            
        if as_text:          
            np.savetxt(par_filename + '_w', self.weights,allow_pickle=True,dtype=object)
            np.savetxt(par_filename + '_b', self.bias,allow_pickle=True,dtype=object)
            
        print(f'Training successful in {elapsed_learning_time}. Parameters saved to:' \
              '{par_filename}_w.npy and {par_filename}_b.npy')
        
        return avg_cost_tracking, avg_eta_tracking if dia_data else None