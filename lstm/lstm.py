import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.utils.data as data
from torchvision import transforms as tfs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import sys
sys.path.append('..')
import datetime

# Create LSTM Model
class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.num_classes = 5 #number of classes
        self.num_layers = 4 #number of layers
        self.input_size = 36 #input size
        self.hidden_size = 100 #hidden state
        self.seq_length = 1 #sequence length

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        #self.fc_1 =  nn.Linear(self.hidden_size, 128) #fully connected 1
        self.fc_1 =  nn.Linear(self.hidden_size, self.num_classes)
        self.relu = nn.ReLU()
        #self.fc = nn.Linear(128, self.num_classes) #fully connected last layer
        #self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)


    def forward(self,x):

        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #internal state
        else:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM

        out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #out = self.fc_1(out) #first Dense
        out = self.relu(hn) #relu
        out = self.fc_1(out) #Final Output

        return out



# Function to save the model
def saveModel():
    path = "./LSTM_Model.pth"
    torch.save(LSTM_model.state_dict(), path)

def fit_model(LSTM_model, loss_func, LSTM_optimizer, num_epochs, train_loader, test_loader, input_shape):
    # Traning the Model
    #history-like list for store loss & acc value
    best_accuracy = 0.0
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1.Define variables
            train = Variable(images.view(input_shape))
            labels = Variable(labels)

            if torch.cuda.is_available():
                train = train.cuda()
                labels = labels.cuda()
                LSTM_model = LSTM_model.cuda()
            # 2.Clear gradients
            LSTM_optimizer.zero_grad()
            # 3.Forward propagation
            outputs = LSTM_model(train)
            if torch.cuda.is_available():
                outputs = outputs.cpu()
                labels = labels.cpu()
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            LSTM_optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()

        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        LSTM_model.train()
        for i,(images, labels) in enumerate(test_loader):
            # 1.Define variables
            test = Variable(images.view(input_shape))
            labels = Variable(labels)
            #print(test.shape)

            if torch.cuda.is_available():
                test = test.cuda()
                labels = labels.cuda()
                LSTM_model = LSTM_model.cuda()

            # 2.Forward propagation
            outputs = LSTM_model(test)
            if torch.cuda.is_available():
                outputs = outputs.cpu()
                labels = labels.cpu()
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print("===== Train Epoch %i/%i =====" % (epoch+1,num_epochs))
        print('Traing_Loss: {} Val_Loss: {} Traing_acc: {:.6f}% Val_accuracy: {:.6f}%'.format(train_loss.data, val_loss.data, train_accuracy, val_accuracy))
        if val_accuracy >= best_accuracy:
            saveModel()
            best_accuracy = val_accuracy
    return training_loss, training_accuracy, validation_loss, validation_accuracy

def predict(model, device, data):

    model.eval()     # Enter Evaluation Mode
    with torch.no_grad():
        pred = model(data.to(device))
        pred = pred.cpu()
        return pred
if __name__ == "__main__":
    start = datetime.datetime.now()

    pic_size = 36
    image_path = './dataset_lstm/'

    # 印出dataset中各類有幾張tensor
    for image_count in os.listdir(image_path):
        print(str(len(os.listdir(image_path + image_count))) + " " + image_count + " tensor")

    # 記錄總共有幾張tensor
    file_count = 0
    for floderName in os.listdir(image_path):
        for filename in os.listdir(image_path + floderName):
            file_count +=1
    print('all_tensor_file: ',file_count)

    label_default = np.zeros(shape=[file_count])
    img_default = np.zeros(shape=[file_count,1,pic_size])
    file_count = 0
    #tensor = []
    #tensors = []
    for floderName in os.listdir(image_path):
        number=0
        for filename in os.listdir(image_path + floderName):
            number+=1

            new_tensor_data=np.load(image_path + floderName + "/" + filename)
            new_tensor_data=np.reshape(new_tensor_data,(-1,36))

            img_default[file_count] = new_tensor_data

            if floderName == '0':
                label_default[file_count] = 0
            elif floderName == '1':
                label_default[file_count] = 1
            elif floderName == '2':
                label_default[file_count] = 2
            elif floderName == '3':
                label_default[file_count] = 3
            elif floderName == '4':
                label_default[file_count] = 4
            file_count +=1
    # Pytorch train and test TensorDataset
    # Hyper Parameters
    # batch_size, epoch and iteration
    batch_size = 4
    num_epochs = 50

    # reshape成丟進model input的dimension
    img_default = img_default.reshape(file_count,1,pic_size,1)
    img_default.shape

    #label_onehot=to_categorical(label_default) # 做onehot encoding
    label_onehot=label_default # 不做onehot encoding
    print('label_onehot[0]:{},label_dim:{},shape:{}'.format(label_onehot[0],label_onehot.ndim,label_onehot.shape)) # Label(Encoding結果 , 維度, shape)
    img_default = img_default.astype('float32') / 255.0 # 做 normalization

    random_seed  = 42 # 隨機分割
    features_train, features_test, targets_train, targets_test = train_test_split(img_default, label_onehot, test_size = 0.2, random_state=random_seed) # 切分訓練及測試集
    features_train = np.delete(features_train,[0,1,2,3],axis=0)
    targets_train = np.delete(targets_train,[0,1,2,3],axis=0)
    features_test = np.delete(features_test,[0,1,2,3,4,5],axis=0)
    targets_test = np.delete(targets_test,[0,1,2,3,4,5],axis=0)
    print('x_train.shape:{}\n,y_train.shape:{}\nx_test.shape:{}\ny_test.shape:{}'.format(features_train.shape, targets_train.shape, features_test.shape, targets_test.shape)) #(train_img, train_label, test_img, test_label)

    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

    # Pytorch train and test TensorDataset
    train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    # Pytorch DataLoader
    train_loader = data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = data.DataLoader(test, batch_size = batch_size, shuffle = True)


    # Print statistics
    print("Total: ", len(img_default))
    print("Training Set: ", len(train))
    print("Validation Set: ", len(test))

    #LSTM_model = LSTM_Model(input_size, hidden_size, num_classes)
    LSTM_model = LSTM_Model()
    #LSTM_model = LSTM_model.float()
    LR = 0.001
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    print(LSTM_model)
    LSTM_optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=LR)   # optimize all cnn parameters

    input_shape = (-1,4,36)


    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(LSTM_model, loss_func, LSTM_optimizer, num_epochs, train_loader, test_loader, input_shape)
    end = datetime.datetime.now()
    print("time:",end - start)
    print("max_training_loss:",max(training_loss))
    print("max_training_accuracy:",max(training_accuracy))
    print("max_validation_los:",max(validation_loss))
    print("max_tvalidation_accuracy:",max(validation_accuracy))
    print("min_training_loss:",min(training_loss))
    print("min_training_accuracy:",min(training_accuracy))
    print("min_validation_loss:",min(validation_loss))
    print("min_validation_accuracy:",min(validation_accuracy))
    # visualization
    plt.plot(range(num_epochs), training_loss, label='Training_loss', color="blue")
    plt.plot(range(num_epochs), validation_loss, label='validation_loss', color="red")
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./lstm_loss.jpg')
    plt.close()

    plt.plot(range(num_epochs), training_accuracy, label='Training_accuracy', color="blue")
    plt.plot(range(num_epochs), validation_accuracy, label='Validation_accuracy', color="red")
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./lstm_accuracy.jpg')
    plt.close()
