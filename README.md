## 夏旻版本
### 1.產生資料集指令
```python
python generate_file noniid - dir 
```
generate_file: 根據要產生的資料集的檔案
- generate_mnist.py -> 產生 MNIST
- generate_fnist.py -> 產生 FASHION-MNIST
- generate_cifar10.py -> 產生 CIFAR10
- generate_cifar100.py -> 產生 CIFAR100

generate_file 參數修改:
- 如要修改 client 數，需要修改 generate_file 內的 num_clients
- 如要修改切割好的資料集檔案名稱，需要修改 generate_file 內的 dir_path
- 如要修改非獨立同分布的 alpha 值，需修改 utils/dataset_utils.py 內的 alpha

niid: 
- 獨立同分布: False
- 非獨立同分布: noniid

balance:
- non-IID: False
- iid: balance

partition 分布方式:
- Dirichlet distribution: dir
- pat

自己的筆記：
- 在 generate_cifar100.py 中，每個 client 至少分配到 20 個類別
- client 在每個類別至少有幾筆資料，取決於資料集的分佈和 alpha 值，每個類別逐一分配每個 client 擁有某個類別的資料比例
- 每個 client 至少有 40 筆資料 (least_samples = batch_size / (1-train_size))
- training set 和 test set 比例分別是 75% 和 25%，取四捨五入

### 2.執行主程式
```python
python main.py -data Cifar10_alpha01_100 -nc 100 -jr 0.5 -algo FedUCBN -sca UCB -gr 499 -pr 0.4 -nb 10
```

\-data 選擇欲執行的資料集:
- MNIST: mnist_alpha01_100、mnist_alpha05_100
- FASHION-MNIST: fmnist_alpha01_100、fmnist_alpha05_100
- CIFAR10: Cifar10_alpha01_100、Cifar10_alpha05_100、Cifar10_alpha10_100
- CIFAR100: Cifar100_alpha01_100、Cifar100_alpha05_100

\-nc 客戶端總數

\-jr 每回合參與訓練比例

\-algo Server聚合演算法
- FedUCBN
- FedAvg

\-sca 客戶端選擇演算法
- Random
- UCB
- UCB_cs (預設 reward 為 accuracy，reward 調整為 loss 需調整 myserver.py 和 serverbase.py)

\-gr 訓練總回合數

\-pr 中毒率，0.0 為無攻擊，0.4 為 40% 中毒率

\-nb 資料集總類別數

\-lr 本地模型學習率，預設 0.005

\-ls 本地訓練回合數 (local epoch)

\-lbs 本地 batch size

### 3.模型
目前使用 code/system/flcore/trainmodel/models.py 的 FedAvgCNN 類別

### 4.客戶端選擇策略
在 code/system/flcore/servers/client_selection/ 內新增修改
- Random
- UCB
- UCB_cs

### 5.Server聚合演算法
在 code/system/flcore/servers/ 內新增修改，基於 serverbase.py 進行修改

### 6.Attack
目前採用 label flippling attack
把標籤 1 換成 9

code/system/flcore/servers/serverbase.py
在function select_poisoned_client 中，選取哪個 client 是會被汙染的

code/system/flcore/clients/clientbase.py
在 function load_train_data 中，將惡意 client 的資料的 1 label 換成 9

### 補充
如要修改最後檔案儲存的位置需要在 serverbase.py 調整


****
## 棨翔版本
### 1.注意事項

1. 強烈建議建一個python的虛擬環境來跑實驗 (怕套件跟自己電腦套件衝突引發大爆炸的話)!
2. 我目前已經修正並新增學長在requirement.txt中缺少列出的套件
3. 學長的code/utils/data_utils.py存在路徑上的錯誤，這部分已修正
4. 修正`python main.py`無法執行的問題，原因是沒有去執行dataset/generate_mnist.py (我這邊使用的sys args是noniid balance dir)

### 2.使用說明

1. 建好並啟動python虛擬環境 (強烈建議!!!)
2. 在命令提示字元執行`pip install -r requirements.txt`
3. 安裝完套件後切換路徑到code/system
4. 在命令提示字元執行`python main.py` (沒輸入參數的情況下會按照預設值跑，詳細資訊請參考main.py)
5. 正常來說，命令提示字元的視窗內會開始跑2000輪的訓練 (會跑很久...我家裡的電腦比較舊會占用很多CPU資源，沒獨顯只有內顯QQ)

(建議只是測試不要跑滿2000輪，可以到main.py裡面改預設值!)

### 3.已知錯誤 (在其他電腦上) -> 2024/07 已解決 (修改路徑即可) by 夏旻

1. OSError: [WinError 126] (暫時無解，我只知道是缺少torch需要的.dll檔，但不知道是哪個)
2. 無法儲存實驗跑完結果 (有解但未解，問題出在路徑不存在)
![](https://i.imgur.com/y7oxlVR.png)

### 4.看完部分程式碼後的建議

1. 可以新增early stopping功能

### 5.執行過程範例影片

[影片連結](https://www.youtube.com/embed/GGPfxRIfAWY?si=wSSsqsiAZZHvhKOK)

---

#### 特別感謝原始碼提供者 : 鍾明翰 學長 [學長筆記連結](https://hackmd.io/XyJWVGecSRWu4jn0haT8mg)
#### 2024.06.28 編輯 by 棨翔 [Federated Learning 學長交接討論重點摘要](https://hackmd.io/@qixiang1009/BkubNnkj6)
#### 2025.03.27 編輯 by 夏旻 改編自 2024.06.28 棨翔的程式碼

