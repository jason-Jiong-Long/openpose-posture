# openpose-posture
## 感謝[Hzzone](https://github.com/Hzzone)貢獻pytorch-openpose程式和安裝說明
### 目的是運用深度學習完成姿勢辨識，採集人體關節運用[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Openpose辨識人體骨骼關節，獲得關節座標，並且將公車乘客分別正常等公車、正常舉手搭公車、輪椅等公車、輪椅舉手搭公車，將一個人18個關節座標收集與分類成四種資料集，藉由資料建立LSTM訓練模型，並將LSTM的layers設定為10，在最後的結果做神經元運算，完成姿勢模型訓練和辨識

## pytorch-openpose安裝執行說明:
### 1.下載[pytorch-openpose程式](https://github.com/Hzzone/pytorch-openpose)，並放入電腦路徑下
### 2.利用[Anaconda](https://www.anaconda.com/products/distribution)建立虛擬環境
開啟Anaconda Prompt:
  
    conda create -n pytorch-openpose python=3.7

### 3.進入虛擬環境[詳細Anaconda虛擬環境操作說明](https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566)

執行程式可切換虛擬環境，防止造成很雜亂的環境，而影響其他程式執行:

  
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
### 7.將openpose訓練資料放入modle資料夾
[下載openpose訓練資料](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG)
### 8.執行openpose(即時)
請確認在pytorch-openpose路徑和環境底下執行，並準備攝像頭:

    python demo_camera.py

### 9.執行openpose(圖片)
準備圖片放入images資料夾中
修改demo.py的程式，改成要跑的圖片名稱(請注意檔名"demo"和副檔名*.jpg):

    test_image = 'images/demo.jpg'

請確認在pytorch-openpose路徑和環境底下執行:

    python demo.py

### 10.執行openpose(影片)
準備影片放入video資料夾中
請確認在pytorch-openpose路徑和環境底下執行:

    ffmpeg -i video/輸入影片名稱.mp4 -vf scale=800:400 video/output.mp4

執行程式:

    python demo_video.py video/output.mp4

## ann訓練執行說明:
### 1.將openpose訓練資料放入modle資料夾
[下載openpose訓練資料](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG)
### 2.收集資料
請在ann路徑底下執行
將辨識的姿勢錄製成一段影片，並將影片放入ann/video資料夾中

    ffmpeg -i video/輸入影片名稱.mp4 -vf scale=800:400 video/output.mp4

執行程式:

    python demo_video_data.py video/output.mp4

**注意收集資料的路徑和檔案名稱，要從demo_video_data.py修改程式碼**
### 3.訓練ann
請在ann路徑底下執行
執行:

    pyton ann.py

### 4.辨識姿勢(即時)
請確認在ann路徑和pytorch-openpose環境底下執行，並準備攝像頭:

    python ann_predict_camera.py

### 5.辨識姿勢(圖片)
準備圖片放入ann/images資料夾中
修改ann_predict.py的程式，改成要跑的圖片名稱(請注意檔名"demo"和副檔名*.jpg):

    test_image = 'images/demo.jpg'

請確認在ann路徑和pytorch-openpose環境底下執行:

    python ann_predict.py

### 6.辨識姿勢(影片)
準備影片放入ann/video資料夾中
請確認在ann路徑和pytorch-openpose環境底下執行:

    ffmpeg -i video/輸入影片名稱.mp4 -vf scale=800:400 video/output.mp4

執行程式:

    python ann_predict_video.py video/output.mp4

## lstm訓練執行說明:
### 1.將openpose訓練資料放入modle資料夾
[下載openpose訓練資料](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG)
### 2.收集資料
請在lstm路徑底下執行
將辨識的姿勢錄製成一段影片，並將影片放入lstm/video資料夾中

    ffmpeg -i video/輸入影片名稱.mp4 -vf scale=800:400 video/output.mp4

執行程式:

    python demo_video_data.py video/output.mp4

**注意收集資料的路徑和檔案名稱，要從demo_video_data.py修改程式碼**
### 3.訓練lstm
請在lstm路徑底下執行
執行:

    pyton lstm.py

### 4.辨識姿勢(即時)
請確認在lstm路徑和pytorch-openpose環境底下執行，並準備攝像頭:

    python lstm_predict_camera.py

### 5.辨識姿勢(圖片)
準備圖片放入lstm/images資料夾中
修改ann_predict.py的程式，改成要跑的圖片名稱(請注意檔名"demo"和副檔名*.jpg):

    test_image = 'images/demo.jpg'

請確認在lstm路徑和pytorch-openpose環境底下執行:

    python lstm_predict.py

### 6.辨識姿勢(影片)
準備影片放入lstm/video資料夾中
請確認在lstm路徑和pytorch-openpose環境底下執行:

    ffmpeg -i video/輸入影片名稱.mp4 -vf scale=800:400 video/output.mp4

執行程式:

    python lstm_predict_video.py video/output.mp4
