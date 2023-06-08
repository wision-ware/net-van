# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:49:56 2023

@author: vavri
"""

import numpy as np
import time
import os
from glob import glob
from decimal import Decimal

class Learn_network(object):

    path_ = os.path.join('..','..','training_params')

    def ReLU(x, d=False):

        return np.where(x<0,0.,(x if d==False else 1))


    def Sigmoid(x, d=False):

        return (1/(1+np.exp(-x))) if d == False\
        else ((1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))))


    def typeval_assertion(t_condition,v_condition,t_message,v_message):

        try:
            assert t_condition
        except AssertionError:
            raise TypeError(t_message)

        try:
            assert v_condition
        except AssertionError:
            raise ValueError(v_message)


    def extract_int(string, cut=None):

        if cut is None:
            n_str = ''
            for i, char in enumerate(string):
                n_str += char if char.isdigit() == True else ''
            n_str = int(n_str)

        else:
            try:
                assert cut in ['first','last'], "Split needs to be either \"first\" or \"last\""
            except AssertionError:
                raise

            if cut == "first":
                n_str = ''
                for i, char in enumerate(string):
                    n_str += char if char.isdigit() == True else ''
                    if char == '_': break

            if cut == "last":
                n_str = ''
                for i,char1 in enumerate(string):
                    if char1 == '_':
                        for char2 in string[i:]:
                            n_str += char2 if char2.isdigit() == True else ''
                        break
        return int(n_str)


    def clear_dir(indices=None):

        try:
            path_ = str(np.copy(Learn_network.path_))
            if indices is not None:

                files = []

                for n in indices:
                    n = int(n)
                    files_ = glob(os.path.join(path_,f'p{n}*.npy'))
                    files.extend(files_)

                for file in files:
                    os.remove(file)

            else:
                files = glob(os.path.join(path_,'p*.npy'))
                for file in files:
                    os.remove(file)

        except (ValueError, IndexError):
            print('Warning: Input indices must correspond with existing parameter files!')


    def current_index():

        path_ = str(np.copy(Learn_network.path_))
        all_files = glob(os.path.join(path_,'p*.npy'))
        if all_files != []:
            end_basename = os.path.basename(all_files[-1])
            extracted = Learn_network.extract_int(end_basename,cut='first')
            return extracted
        else: return 0

    def __init__(self, N):

        # verifying parameters

        confirmation = [(True if isinstance(i,int) else False) for i in N]
        Learn_network.typeval_assertion(
            isinstance(N, (np.ndarray, list)),
            all(confirmation),
            f"positional argument \'N\' must be type: \'list\' or \'numpy.ndarray\', not {type(N)}!",
            "all of the elements of positional argument \'N\' must be type: \'int\'!"
            )

        # --

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

        path_ = str(np.copy(Learn_network.path_))
        dir_ind = Learn_network.current_index()
        self.par_filename = os.path.join(path_,f'p{dir_ind}')


    def get_output(self, inp, layer=False, label=None):

        # verifying  parameters

        Learn_network.typeval_assertion( # training data verification
            type(inp) == np.ndarray,
            len(inp.shape) == 2,
            f"positional argument \'inp\' must be type: \'numpy.ndarray\', not {type(inp)}!",
            f"positional argument \'inp\' must be 2 dimensional (samples, data_width), {len(inp.shape)} dimensional was given!"
            )
        try:
            assert inp.shape[1] == self.N[0]
        except AssertionError:
            raise ValueError(f"size of the second dimension of the positional argument \'inp\' must be equal to the number of input nodes of the first layer! ({self.N[0]} required, {inp.shape[1]} given)")

        Learn_network.typeval_assertion(
            isinstance(layer, bool) or isinstance(layer, int),
            isinstance(layer, bool) or layer >= 0,
            f"keyword argument \'layer\' must be type \'int\' or \'bool\', not {type(layer)}!",
            "keyword argument \'layer\' can not be negative!"
            )

        # --

        if layer == True:
            all_out_act = []
            all_out = []

        # first layer output

        p_output = np.copy(inp)
        if layer == True:
            all_out_act.append(p_output[:])
            all_out.append(p_output[:])

        # rest of the layers propagating

        if (layer > 1) or (len(self.N) > 1):

            for l in range(1,(len(self.N) if type(layer) == bool else layer)):

                act_matrix = np.full((self.N[l],self.N[l-1]),p_output)
                act_matrix = np.transpose(act_matrix)*self.weights[l]
                activation = np.sum(act_matrix,axis=0) + self.bias[l]

                if l < (len(self.N)-1): p_output = Learn_network.ReLU(activation)
                else: p_output = Learn_network.Sigmoid(activation)

                if layer == True:
                    all_out_act.append(activation[:])
                    all_out.append(p_output[:])

        # computing cost

        if label is not None:

            if layer==False or type(layer)==int: output = np.copy(p_output)
            else: output = np.copy(all_out[-1])

            dif = label - output
            cost = np.sum(dif**2)
            self.cost = cost

        if layer == True: return [all_out_act[:],all_out[:]]
        else: return np.copy(p_output)

    # backpropagation

    def backpropagate(self, inp, labels):

        # verifying  parameters

        Learn_network.typeval_assertion( # training data verification
            type(inp) == np.ndarray,
            len(inp.shape) == 2,
            f"positional argument \'inp\' must be type: numpy.ndarray, not {type(inp)}!",
            f"positional argument \'inp\' must be 2 dimensional (samples, data_width), {len(inp.shape)} dimensional was given!"
            )
        try:
            assert inp.shape[1] == self.N[0]
        except AssertionError:
            raise ValueError(f"size of the second dimension of the positional argument \'inp\' must be equal to the number of input nodes of the first layer! ({self.N[0]} required, {inp.shape[1]} given)")

        Learn_network.typeval_assertion( # data label verification
            type(labels) == np.ndarray,
            len(labels.shape) == 2,
            f"positional argument \'labels\' must be type: numpy.ndarray, not {type(inp)}!",
            f"positional argument \'labels\' must be 2 dimensional (samples, binary_sort_cases), {len(inp.shape)} dimensional was given!"
            )
        try:
            assert labels.shape[1] == self.N[-1]
        except AssertionError:
            raise ValueError(f"size of the second dimension of the positional argument \'labels\' must be equal to the number of output nodes of the final layer! ({self.N[-1]} required, {labels.shape[1]} given)")

        # --

        gradient = self.weight_like[:]
        partial_bias = self.bias_like[:]
        output = self.get_output(inp,layer=True,label=labels)

        # output layer

        dsigmoid = Learn_network.Sigmoid(output[0][-1],True)
        dif = output[1][-1]-labels[:]
        deltas = dsigmoid*dif
        partial_bias_0 = deltas[:]

        m_deltas = np.full((self.N[-2],self.N[-1]),deltas)
        m_output = np.full((self.N[-1],self.N[-2]),output[1][-2])
        grad_0 = m_deltas*np.transpose(m_output)

        gradient[-1] = grad_0[:]
        partial_bias[-1] = partial_bias_0[:]
        deltas_old = deltas[:]

        # hidden layers

        for l in range(2,len(self.N)):

            drelu = Learn_network.ReLU(output[0][-l][:],True)
            m_drelu = np.full((self.N[-(l-1)],self.N[-l]),drelu)
            m_deltas_old = np.full((self.N[-l],self.N[-(l-1)]),deltas_old)

            summaries = np.transpose(m_drelu)*m_deltas_old*self.weights[-(l-1)]
            deltas_new = np.sum(summaries,axis=1)[:]

            m_deltas_new = np.full((self.N[-(l+1)],self.N[-l]),deltas_new)
            m_output = np.full((self.N[-l],self.N[-(l+1)]),output[1][-(l+1)])

            gradient[-l] = m_deltas_new*np.transpose(m_output)
            partial_bias[-l] = deltas_new[:]
            deltas_old = deltas_new[:]

        return [gradient[:],partial_bias[:]]

    # learning algorithm with optional learning rate, cost treshold and GD methods

    def learn(
            self,
            inp,
            labels,
            treshold=1e-12,
            time_limit=np.infty,
            GD='mini_b',
            batch_size=50,
            eta=0.005,
            live_monitor=False,
            as_text=False,
            fixed_iter=0,
            dia_data=False,
            save_params=True,
            overwrite=True
            ):

        # verifying  parameters

        Learn_network.typeval_assertion( # training data verification
            type(inp) == np.ndarray,
            len(inp.shape) == 2,
            f"positional argument \'inp\' must be type: numpy.ndarray, not {type(inp)}!",
            f"positional argument \'inp\' must be 2 dimensional (samples, data_width), {len(inp.shape)} dimensional was given!"
            )
        try:
            assert inp.shape[1] == self.N[0]
        except AssertionError:
            raise ValueError(f"size of the second dimension of the positional argument \'inp\' must be equal to the number of input nodes of the first layer! ({self.N[0]} required, {inp.shape[1]} given)")

        Learn_network.typeval_assertion( # data label verification
            type(labels) == np.ndarray,
            len(labels.shape) == 2,
            f"positional argument \'labels\' must be type: numpy.ndarray, not {type(inp)}!",
            f"positional argument \'labels\' must be 2 dimensional (samples, binary_sort_cases), {len(inp.shape)} dimensional was given!"
            )
        try:
            assert labels.shape[1] == self.N[-1]
        except AssertionError:
            raise ValueError(f"size of the second dimension of the positional argument \'labels\' must be equal to the number of output nodes of the final layer! ({self.N[-1]} required, {labels.shape[1]} given)")

        Learn_network.typeval_assertion( # cost treshold verification
            isinstance(treshold,(float,Decimal)),
            treshold > 0,
            f"keyword argument \'treshold\' must be a number, not {type(treshold)}!",
            "keyword argument \'treshold\' must be positive!"
            )
        Learn_network.typeval_assertion( # training time limit verification
            isinstance(time_limit,(float,Decimal)),
            time_limit > 0,
            f"keyword argument \'time_limit\' must be a number, not {type(treshold)}!",
            "keyword argument \'time_limit\' must be positive!"
            )
        GD_options = ['mini_b','batch','stochastic'] # gradient descent type switch options
        Learn_network.typeval_assertion( # gradient descent type switch verification
            isinstance(GD,str),
            GD in GD_options,
            f"keyword argument \'GD\' must be type \'str\', not {type(GD)}!",
            f"keyword argument \'GD\' must be one of the following: {GD_options}"
            )
        Learn_network.typeval_assertion( # batch size verification
            isinstance(batch_size,int),
            batch_size > 0,
            f"keyword argument \'batch_size\' must be type \'int\', not {type(batch_size)}!",
            "keyword argument \'batch_size\' must be positive!"
            )
        Learn_network.typeval_assertion( # batch size verification
            isinstance(eta,(float,Decimal)),
            eta > 0,
            f"keyword argument \'eta\' must be type \'int\', not {type(eta)}!",
            "keyword argument \'eta\' must be positive!"
            )
        try: # live monitor toggle verification
            assert isinstance(live_monitor,bool)
        except AssertionError:
            raise TypeError(f"keyword argument \'live_monitor\' must be type \'bool\', not {type(live_monitor)}!")

        try: # saving in .txt format toggle verification
            assert isinstance(as_text,bool)
        except AssertionError:
            raise TypeError(f"keyword argument \'as_text\' must be type \'bool\', not {type(as_text)}!")

        Learn_network.typeval_assertion( # verification of the fixed number of iterations
            isinstance(fixed_iter,int),
            fixed_iter >= 0,
            f"keyword argument \'fixed_iter\' must be type \'int\', not {type(fixed_iter)}!",
            "keyword argument \'fixed_iter\' can not be negative!"
            )
        try: # diagnostic data return toggle verification
            assert isinstance(dia_data,bool)
        except AssertionError:
            raise TypeError(f"keyword argument \'dia_data\' must be type \'bool\', not {type(dia_data)}!")

        try: # trained parameter saving to binary (.npy) format verification
            assert isinstance(save_params,bool)
        except AssertionError:
            raise TypeError(f"keyword argument \'save_params\' must be type \'bool\', not {type(save_params)}!")

        Learn_network.typeval_assertion(
            isinstance(overwrite, bool) or isinstance(overwrite, int),
            isinstance(overwrite, bool) or overwrite >= 0,
            f"keyword argument \'overwrite\' must be type \'int\' or \'bool\', not {type(overwrite)}!",
            "keyword argument \'overwrite\' can not be negative!"
            )

        # --

        d_index = 0
        gamma = 0.9
        eta_r = 0.001
        self.get_output(inp[0,:],False,label=labels[0,:])
        avg_cost = 999

        if dia_data or live_monitor: avg_cost_tracking = []
        if live_monitor: empty_chars = ""
        if dia_data: avg_eta_tracking = []

        if GD == 'mini_b':
            R_w = self.weight_like[:]
            R_b = self.bias_like[:]

        dif_w = self.weight_like[:]
        dif_b = self.bias_like[:]

    # parameter init

        for l in range(1,len(self.N)):
            self.weights[l] = np.random.normal(
            0,2/np.sqrt(self.N[l] + self.N[l-1]),(self.N[l-1],self.N[l]))\
            if l < (self.N[-1]) else np.random.normal(
            0,np.sqrt(2/(self.N[l] + self.N[l-1])),(self.N[l-1],self.N[l])
            )

        t_0 = time.process_time()
        elapsed_learning_time = 0

    # main training loop

        while (d_index < fixed_iter) if fixed_iter != 0\
            else ((avg_cost > treshold) & (elapsed_learning_time < time_limit)):

            if GD == 'stochastic':
                d_indices = np.arange(len(inp))
                ind = np.random.choice(d_indices)
                partials = self.backpropagate(inp[ind,:],labels[ind,:])
                avg_cost = self.cost/self.N[-1]

            if GD == 'batch':

                s_partials = [self.weight_like[:],self.bias_like[:]]
                iter_cost_sum = 0

                for i in range(len(inp)):

                    backprop_out = self.backpropagate(inp[i,:],labels[i,:])
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
                    backprop_out = self.backpropagate(inp[ind,:],labels[ind,:])
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

            if live_monitor or dia_data: avg_cost_tracking.append(avg_cost)
            if live_monitor:
                print(empty_chars,end='\r')
                message = 'Current cost minimum: ' + str(min(avg_cost_tracking))
                print(message,end='\r')
                empty_chars = "\b"*(len(message))

            if dia_data:
                avg_eta_tracking_ = np.average(np.array(avg_eta_tracking_))
                avg_eta_tracking.append(avg_eta_tracking_)

        path_ = str(np.copy(Learn_network.path_))
        if live_monitor:
            empty_chars = "\b"*(len(message))
            print(empty_chars,end='\r')

        if type(overwrite) == bool:

            dir_content = glob(os.path.join(path_,'p*.npy'))

            if dir_content != []:
                current_ind = Learn_network.current_index()
            else:
                current_ind = 0

            if overwrite:
                deletions = glob(os.path.join(path_,f'p{current_ind}*.npy'))
                for file in deletions:
                    os.remove(file)
                saving_filename = os.path.join(path_,f'p{current_ind}')
            else:
                saving_filename = os.path.join(path_,f'p{current_ind}')

        elif type(overwrite) == int:
            saving_filename = os.path.join(path_,f'p{overwrite}')
            deletions = glob(os.path.join(path_,f'p{overwrite}*.npy'))
            for file in deletions:
                os.remove(file)

        else:
            raise TypeError('overwrite argument accepts only type bool and integer')

        if save_params:
            for l in range(1,len(self.N)):
                np.save(saving_filename + '_w' + str(l-1), self.weights[l],allow_pickle=True)
                np.save(saving_filename + '_b' + str(l-1), self.bias[l],allow_pickle=True)

        if as_text:
            np.savetxt(saving_filename + '_w', self.weights,allow_pickle=True)
            np.savetxt(saving_filename + '_b', self.bias,allow_pickle=True)

        print(f'Training successful after {elapsed_learning_time}s.',end='\n')

        return_dict = {'weights':self.weights,
                       'bias':self.bias}

        if dia_data:
            return_dict['cost'] = avg_cost_tracking
            return_dict['l_rate'] = avg_eta_tracking

        return return_dict







