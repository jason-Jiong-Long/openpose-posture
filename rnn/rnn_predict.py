import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
from torch import nn

from src import model
from src import util
from src.body import Body
from src.hand import Hand
from lstm import LSTM_Model
from torch.autograd import Variable

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
data_name=0

test_image = 'images/abc.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)


def testBatch(lstmdata):
    import torchvision
    '''
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                               for j in range(batch_size)))
    '''
    # Let's see what if the model identifiers the  labels of those example

    if torch.cuda.is_available():
        lstmdata = torch.tensor(lstmdata).cuda()
    else:
        print("請使用GPU")

    outputs = LSTM_Model(lstmdata.float())

    if torch.cuda.is_available():
        outputs = outputs.cpu()
    else:
        print("請使用GPU")

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    predicted = np.array(predicted)
    print(predicted)
    for i in predicted:
        if i==0:
            print("正常")
        elif i ==1:
            print("跌倒")
        elif i ==2:
            print("舉手")
        elif i ==3:
            print("行動不便舉手")
        elif i ==4:
            print("行動不便")
    # Let's show the predicted labels on the screen to compare with the real ones
    #print('Predicted: ', ' ',predicted)
    #print(type(predicted))

def json_data(candidate, subset):
    global data_name
    '''
    print(type(candidate))
    print(type(subset))
    print(candidate.shape)
    print(subset.shape)
    print(candidate)
    print(candidate[5,3])
    print(subset)
    '''
    for a in range(subset.shape[0]):
        data_dict={}
        #print(a)
        for b in range(18):
            for c in range(candidate.shape[0]):
                if subset[a,b]==int(candidate[c,3]):
                    data_dict.setdefault(b,[candidate[c,0],candidate[c,1],candidate[c,2]])

                if subset[a,b]==-1:
                    data_dict.setdefault(b,[-1,-1,-1])


        data_list1=[]
        data_list2=[]
        data_list3=[]
        data_list4=[]
        data_list5=[]
        data_list6=[]
        data_list7=[]
        data_list8=[]
        data_list9=[]
        data_list10=[]
        data_list11=[]
        data_list12=[]
        data_list13=[]
        data_list14=[]
        data_list15=[]
        data_list16=[]
        data_list17=[]
        data_list18=[]

        for d in range(6):
            data_list1.append(data_dict[d][0])
        for e in range(6):
            data_list2.append(data_dict[e+6][0])
        for f in range(6):
            data_list3.append(data_dict[f+12][0])
        for g in range(6):
            data_list4.append(data_dict[g][1])
        for h in range(6):
            data_list5.append(data_dict[h+6][1])
        for i in range(6):
            data_list6.append(data_dict[i+12][1])
        data_lists1=np.vstack([data_list1,data_list2,data_list3,data_list4,data_list5,data_list6])

        for j in range(6):
            data_list7.append(data_dict[j][1])
        for k in range(6):
            data_list8.append(data_dict[k+6][1])
        for l in range(6):
            data_list9.append(data_dict[l+12][1])
        for m in range(6):
            data_list10.append(data_dict[m][2])
        for n in range(6):
            data_list11.append(data_dict[n+6][2])
        for o in range(6):
            data_list12.append(data_dict[o+12][2])
        data_lists2=np.vstack([data_list7,data_list8,data_list9,data_list10,data_list11,data_list12])

        for p in range(6):
            data_list13.append(data_dict[p][2])
        for q in range(6):
            data_list14.append(data_dict[q+6][2])
        for r in range(6):
            data_list15.append(data_dict[r+12][2])
        for s in range(6):
            data_list16.append(data_dict[s][0])
        for t in range(6):
            data_list17.append(data_dict[t+6][0])
        for u in range(6):
            data_list18.append(data_dict[u+12][0])
        data_lists3=np.vstack([data_list13,data_list14,data_list15,data_list16,data_list17,data_list18])
        data_list_all=np.stack((data_lists1,data_lists2,data_lists3),axis=0)
        '''
        print(data_dict)
        print(data_lists1)
        print(data_lists2)
        print(data_lists3)
        print(data_list_all)
        np.save('./dataset/engine_data'+str(data_name)+'.npy', data_list_all)
        data_name+=1
        #print(data_dict.values())
        with open("json/output"+str(a)+".json", "w") as f:
            json.dump(data_dict, f)
        '''
        return data_lists1





data_teat=json_data(candidate, subset)

import torch
lstm_path = "LSTM_Model.pth"
LSTM_Model=LSTM_Model()

LSTM_Model.load_state_dict(torch.load(lstm_path))
print(data_teat)
data_teat=np.reshape(data_teat,(-1,1,36))
if torch.cuda.is_available():
    LSTM_Model = LSTM_Model.cuda()
else:
    print("請使用GPU")
#print(data_teat)

# Test with batch of images
testBatch(data_teat)


canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)
# detect hand
hands_list = util.handDetect(candidate, subset, oriImg)

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
