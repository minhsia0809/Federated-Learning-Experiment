import torch
import torch.distributions as tdist

import math
import numpy as np
import statistics
import copy


class UCB_cs:
    
    def __init__(self, num_clients, num_join_clients, global_rounds, clients_data_ratio):
        
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.global_rounds = global_rounds
        ## self.all_client_data = sum(c.train_samples for c in self.all_clients)
        ## self.clients_data_ratio = [c.train_samples / self.all_client_data for c in self.all_clients]
        self.clients_data_ratio = clients_data_ratio
        self.each_clients_selections = np.zeros((self.num_clients, self.global_rounds + 1), dtype=int)
        self.local_losses = np.zeros((self.num_clients, self.global_rounds + 1))
        self.gamma = 0.9
        self.tau = 100
        self.now_round = 0
        
        '''
        all_client_data = 0
        all_client_data_ratio = list()
        for c in range(self.clients):
            all_client_data = all_client_data + c.train_data
        for c in range(self.clients):
            all_client_data_ratio.append(c.train_data/all_client_data)
        '''
            
    def get_n_max(self, n, target):
        
        ## return np.argsort(target)[len(target) - n:]
    
        
        t = copy.deepcopy(target)
        # 求m个最大的数值及其索引
        max_index = []
        for _ in range(n):
            index = np.argmax(t)
            t[index] = -1.0
            max_index.append(index)
        return max_index
        

    def select_clients(self, epoch):
        
        clients_upper_bound = list()
        this_round_losses = list()
        
        if epoch < 2:
            
            this_round_losses = [0] * self.num_join_clients
            
        else:
            
            for i in range(self.num_clients):
                if self.local_losses[i][epoch - 1] > 0:
                    this_round_losses.append(self.local_losses[i][epoch - 1])
                    
        for i in range(self.num_clients):
            
            At, Lt, Nt, Tt, Ut = 0, 0, 0, 0, 0
            
            if (sum(self.each_clients_selections[i][:epoch]) > 0):
                
                for tp in range(epoch + 1):
                    g = math.pow(self.gamma, epoch - tp)
                    Lt += g * (1 / self.tau) * self.local_losses[i][tp]
                    Nt += g * self.each_clients_selections[i][tp]
                    Tt += g
                if len(this_round_losses) == 1:
                    this_round_losses.append(this_round_losses[0])
                ## print(this_round_losses)
                Ut = math.sqrt(2 * statistics.variance(this_round_losses) * (math.log(Tt / Nt, 2) ))
                At = self.clients_data_ratio[i] * (Lt / Nt + Ut)
                
            else:
                At = 1e+40
            
            clients_upper_bound.append(At)
        
        selected_clients = self.get_n_max(self.num_join_clients, clients_upper_bound)
        self.now_round = epoch

        return selected_clients

    def update(self, selected_ids, rewards):
        
        reward_decay = 1
        
        for client_id, reward in zip(selected_ids, rewards):
            self.each_clients_selections[client_id][self.now_round] = 1 
            self.local_losses[client_id][self.now_round] = reward