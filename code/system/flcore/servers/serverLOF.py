import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import numpy as np

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.UCB_cs import UCB_cs



class FedLOF(Server):
    def __init__(self, args, times, agent = None):
        super().__init__(args, times)

        # self.args = args
        self.agent = agent
        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def get_vector_no_bn(self, model):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        v = []
        for key in model.state_dict():
            if key in bn_key:
                continue 
            v.append(model.state_dict()[key].view(-1))
        return torch.cat(v)
    
    def get_vector(self, model):
        return parameters_to_vector(model.parameters()).detach()
    
    def train(self):
        start_time = time.time() # <- mhsia
        
        self.send_models() #initialize model
        testloaderfull = self.get_test_data()

        if self.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)

        elif self.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)
        
        elif self.select_clients_algorithm == "UCB_cs":
            all_client_data = sum(c.train_samples for c in self.clients)
            clients_data_ratio = [c.train_samples / all_client_data for c in self.clients]
            print('clients_data_ratio:', clients_data_ratio)
            select_agent = UCB_cs(self.num_clients, self.num_join_clients, self.global_rounds, clients_data_ratio) ## <= mhsia
        
        elif self.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name = f"noniid_wbn_{self.num_clients*self.poisoned_ratio}_same"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds+1):
                s_t = time.time()
                

                # self.send_models()

                # if i%self.eval_gap == 0:
                #     print(f"\n-------------Round number: {i}-------------")
                #     print("\nEvaluate global model")
                #     self.evaluate()

                selected_ids = select_agent.select_clients(i)
                print("selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]
                self.select_clients_his.append(sorted(selected_ids)) ## mhsia

                # self.selected_clients = self.select_clients()
                # s = [c.id for c in self.selected_clients]
                # print(s)

                # => mh code 
                
                '''
                select client by UCB
                '''
                # self.selected_clients = self.select_clients_UCB(i)
                # s = [c.id for c in self.selected_clients]
                # print(s)

                print(f"\n-------------Round number: {i}-------------")

                print(f"history acc: {self.acc_his}")
                # if len(self.acc_his) == 3 and (max(self.acc_his) - min(self.acc_his)) < 0.015:
                #     print("acc convergence!!!")
                #     break
                # if len(self.acc_his) >= 1 and self.acc_his[-1] >= 0.75:
                #     print("acc to the goal!!")
                #     break

                # self.selected_clients = self.select_clients_by_trust()
                # <= mh code 

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]


                self.receive_models()


                '''
                check whether it is melicious node and record
                '''
                model_vectors = [] # => LOF
                
                # -> mhsia
                clients_acc = []
                for client_model, client in zip(self.uploaded_models, self.selected_clients):
                    test_acc, test_num, auc= self.test_metrics_all(client_model, testloaderfull)
                    #print(test_acc/test_num)
                    clients_acc.append(test_acc/test_num)
                    
                    v = self.get_vector(client_model) # => LOF
                    model_vectors.append(v.cpu().numpy()) # => LOF

                clients_acc_weight = list(map(lambda x: x/sum(clients_acc), clients_acc))
                
                model_vectors = np.stack(model_vectors) # => LOF

                reward_decay = 1
                for reward, client in zip(clients_acc, self.selected_clients):
                    self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                    self.numbers_of_selections[client.id] += 1
                
                rewards = clients_acc
                select_agent.update(selected_ids, rewards)
                # <- mhsia
                
                ## => mhsia code
                '''
                clients_loss = []
                for client_model, client in zip(self.uploaded_models, self.selected_clients):
                    test_loss, test_num, auc = self.test_metrics_all(client_model, testloaderfull)
                    #print(test_loss/test_num)
                    clients_loss.append(test_loss/test_num)
                    
                    v = self.get_vector(client_model) # => LOF
                    model_vectors.append(v.cpu().numpy()) # => LOF
                    
                clients_loss_weight = list(map(lambda x: x/sum(clients_loss), clients_loss))
                
                model_vectors = np.stack(model_vectors) # => LOF

                reward_decay = 1
                for reward, client in zip(clients_loss, self.selected_clients):
                    self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                    self.numbers_of_selections[client.id] += 1
                
                rewards = clients_loss
                select_agent.update(selected_ids, rewards)
                '''
                ## <= mhsia code

                same_weight = [1/self.num_join_clients] * self.num_join_clients
                weight = clients_acc_weight # <- mhsia
                ## weight = clients_loss_weight ## mhsia code
                
                # <= mh code 
                
                if self.weight_option == "same":
                    weight = same_weight
                

                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                    
                ## => LOF
                '''
                model_vectors = []
                for model in self.uploaded_models:
                    v = self.get_vector(model)
                    model_vectors.append(v.cpu().numpy())
                model_vectors = np.stack(model_vectors)
                '''
                # 使用 LOF 檢測異常更新
                lof = LocalOutlierFactor(n_neighbors=2, contamination=0.4)  # contamination 可根據中毒比例調整
                pred = lof.fit_predict(model_vectors)
                
                # 排除被判定為異常的 client
                filtered_models = []
                filtered_clients = []
                filtered_weights = []

                potential_malicious_client = []
                for p, model, client, w in zip(pred, self.uploaded_models, self.selected_clients, weight):
                    if p == 1:  # 1 表示 inlier（正常）
                        filtered_models.append(model)
                        filtered_clients.append(client)
                        filtered_weights.append(w)
                    else:
                        potential_malicious_client.append(client.id)
                print(f"{len(potential_malicious_client)} clients are detected as an outlier by LOF and excluded from aggregation.")
                print(potential_malicious_client)
                
                
                # 替換原始模型列表與 client 列表與權重
                self.uploaded_models = filtered_models
                self.selected_clients = filtered_clients
                
                '''
                # 提前一次性轉為 float32 numpy array，並移至 CPU
                model_vectors = torch.stack([self.get_vector(m).detach().cpu() for m in self.uploaded_models]).numpy().astype(np.float32)

                # 使用 LOF 檢測異常更新
                lof = LocalOutlierFactor(n_neighbors=2, contamination=0.4, n_jobs=-1)  # 使用多執行緒加速
                pred = lof.fit_predict(model_vectors)

                # 快速分群
                mask = pred == 1  # 1 表示 inlier

                # 篩選正常模型與權重
                self.uploaded_models = [m for m, keep in zip(self.uploaded_models, mask) if keep]
                self.selected_clients = [c for c, keep in zip(self.selected_clients, mask) if keep]
                filtered_weights = [w for w, keep in zip(weight, mask) if keep]

                potential_malicious_client = [c.id for c, keep in zip(self.selected_clients, mask) if not keep]
                print(f"{len(potential_malicious_client)} clients are detected as outliers by LOF and excluded from aggregation.")
                print(potential_malicious_client)
                ''' 

                # 重新 normalize 權重
                if self.weight_option != "same":
                    total_weight = sum(filtered_weights)
                    if total_weight > 0:
                        weight = [w / total_weight for w in filtered_weights]
                    else:
                        weight = [1 / len(filtered_weights)] * len(filtered_weights)
                else:
                    weight = [1 / len(filtered_weights)] * len(filtered_weights)
                
                ## <= LOF
                self.aggregate_parameters(weight)  # <- mhsia


                ## self.send_models_bn()
                self.send_models()
                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    acc, train_loss, auc = self.evaluate()
                    #acc, train_loss, auc = self.evaluate_trust()
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step = i)
                    mlflow.log_metric("train_loss", train_loss, step = i)

                # => mh code
                '''
                use selected clients to test accuracy
                '''
                # acc_p = 0
                # for client in self.selected_clients:
                #     ct, ns, auc = client.test_metrics()
                #     acc_p += ct/ns
                # acc_p = acc_p / len(self.selected_clients)
                # print(f"acc_p: {acc_p}")
                # <= mh code

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        total_time = round(time.time()-start_time, 2) ## mhsia
        self.save_results(total_time) ## mhsia
        # self.save_results()
        self.save_global_model()




    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [torch.sign(update) for update in agent_updates]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr   
        return sm_of_signs.to(self.device)
