# openpose-posture
## 感謝[Hzzone](https://github.com/Hzzone)貢獻pytorch-openpose程式和安裝說明
### 目的是運用深度學習完成姿勢辨識，採集人體關節運用[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## 說明:
### 1.下載[pytorch-openpose程式](https://github.com/Hzzone/pytorch-openpose)，並放入電腦路徑下
### 2.利用[Anaconda](https://www.anaconda.com/products/distribution)建立虛擬環境
開啟Anaconda Prompt:
  
    conda create -n pytorch-openpose python=3.7

### 3.進入虛擬環境:[詳細Anaconda虛擬環境操作說明](https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566)
  
    conda activate pytorch-openpose

### 4.安裝python套件
我是手動安裝，不會手動安裝的可以執行以下指令"切記執行前要切換到requirements.txt的目錄底下"，如果不會切換目錄請自行google學習:
 
    pip install -r requirements.txt

### 5.安裝ffmpeg:
需下載[github上的ffmpeg](https://github.com/kkroening/ffmpeg-python/tree/master/examples)
在ffmpeg-python資料夾路徑底下執行:

    python setup.py build
    python setup.py install

接著執行:

    conda install ffmpeg

### 6.檢查安裝套件:
查看安裝套件是否成功在虛擬環境中:

    conda list
    pip list
    
如果執行程式時發現無法執行(程式找不到套件)，手動安裝程式:

    conda install 套件名稱
    
or:

    pip install 套件名稱
    
如果無法安裝請google查詢問題(大部分狀況是套件名稱錯誤或conda、pip需要更新或版本支援問題)
### 7.執行openpose(即時)
請確認在pytorch-openpose路徑底下執行，並準備相機:

    python demo_camera.py

### 7.執行openpose(圖片)
準備圖片放入images資料夾中
修改demo.py的程式，改成要跑的圖片名稱(請注意檔名和副檔名):

    test_image = 'images/demo.jpg'

請確認在pytorch-openpose路徑底下執行:

    python demo.py
