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

### 3.已知錯誤 (在其他電腦上)

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

