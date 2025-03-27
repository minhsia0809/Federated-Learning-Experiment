import argparse
import os
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from pathlib import Path


def draw_acc(dataset, num_clients, record_datas):
    plt.figure()
    
    for i in range(0, len(record_datas)):
        plot_label = ''
        colName = str(record_datas[i].columns.values)[2:]

        if colName.split('_')[0] == 'FedAvg':
            if colName.split('_')[1] == 'UCB':
                if colName.split('_')[2] == '0.0':
                    plot_label = 'FedBN without attack'
                elif float(colName.split('_')[2]) > 0.0:
                    plot_label = 'Baseline(FedBN)'
            elif colName.split('_')[1] == 'Random':
                plot_label = 'Random+Trust'
        elif colName.split('_')[0] == 'FedTrimmed':
            plot_label = 'TrimmedMean'
        elif colName.split('_')[0] == 'FedUCBN':
            #print(colName.split('_'))
            if colName.split('_')[1] == 'Random':
                plot_label = 'TBA+Random+Trust'
            else:
                if colName.split('_')[1] == 'notrust':
                    plot_label = 'MyProposed(TBA)'
                else:
                    plot_label = 'TBA+Trust'
                
        #print(plot_label)
        plt.plot(record_datas[i], label=plot_label)
        plt.legend()
        #print(record_datas[i].columns.values)
    
    
    plt.title(str(num_clients) + ' clients with 40% poisoned')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    
    pltName = dataset + '_acc.png'
    plt.savefig(args.dataset + '/' + str(args.num_clients) +'/acc/'+ pltName)
    #plt.savefig(str(args.num_clients) +'/auc/'+ pltName)

def draw_alpha(num_clients, record_datas):
    plt.figure()
    
    for i in range(0, len(record_datas)):
        plot_label = record_datas[i][0].split('_')[1]
        plt.plot(record_datas[i][1], label=plot_label)
        plt.legend()
    plt.title(str(num_clients) + ' clients with different alpha')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    
    pltName = 'Random_trust_acc.png' #TBA_Random_acc TBA_UCB_trust_acc TBA_UCB_notrust_acc
    plt.savefig('./' + pltName)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nc', "--num_clients", type=int, default=100)
    parser.add_argument('-alpha', "--alpha", type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.alpha == False:
        #db = Path(args.dataset)
        folderPath = args.dataset + '/' + str(args.num_clients) + '/acc/'
        csvFile = os.listdir(folderPath)
        
        #print(csvFile)
        
        record = list()
        
        for i in range(0, len(csvFile)):
            if csvFile[i].endswith('.csv') == False:
                continue
            df_datas = pd.read_csv(folderPath + csvFile[i])
            record.append(df_datas)
            #print(df_datas.columns.values)
        #print(record)
        
        draw_acc(args.dataset, args.num_clients, record)
    else:
        allForder = os.listdir('./')
        record = list()
        for forder in allForder:
            tempList = list()
            if forder.count('_') != 2:
                continue
            if 'alpha' in forder.split('_')[1] and '100' in forder.split('_')[2]:
                folderPath = forder + '/' + str(args.num_clients) + '/acc/'
                csvFile = os.listdir(folderPath)
                
                for i in range(0, len(csvFile)):
                    if csvFile[i].startswith('FedAvg_Random') == False: #FedUCBN_Random FedUCBN_UCB FedUCBN_notrust
                        continue
                    df_datas = pd.read_csv(folderPath + csvFile[i])
                    tempList.append(forder)
                    tempList.append(df_datas)
                    record.append(tempList)
        draw_alpha(args.num_clients, record)
                        
        #print(allForder)
        #draw_alpha()