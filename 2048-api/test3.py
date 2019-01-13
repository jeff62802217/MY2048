import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import pickle
import torchvision


FILTERS=128
EPOCH = 100
LR = 0.001
BATCH_SIZE = 200

map_table = {2**i: i for i in range(1, 16)}
map_table[0] = 0
def grid_arr(arr):
    ret = np.zeros(shape=(16,4, 4),dtype = float)
    for r in range(4):
        for c in range(4):
            ret[map_table[arr[r][c]]][r][c] = 1
    return ret

class DealDataset(Data.Dataset):
    def __init__(self):
        
        data_set = [] 
        for i in range(8):
            pick_file = open('training_data%d.pkl'%i, 'rb')
            data_set += pickle.load(pick_file)
            pick_file.close()
            
        self.data_x = []
        self.data_y = []
        for i in range(len(data_set)):
            self.data_x.append(grid_arr(data_set[i][0]))
            self.data_y.append(data_set[i][1])
            
            #if i==0:
           #     self.data_x[0] = grid_arr(data_set[i][0])
           #     self.data_y[0] = grid_arry(data_set[i][1])
           # self.data_x = np.concatenate(np.zeros(shape=(1, 4, 4, 16),dtype = float), axis = 0)
           # self.data_x[i]= grid_arr(data_set[i][0])
           # self.data_x = np.concatenate(np.zeros(shape=(1, 1, 4),dtype = float), axis = 0)
           # self.data_y[i]= grid_arr(data_set[i][1])
        self.len = len(data_set)
        #self.data_x = torch.from_numpy(self.data_x)
        #self.data_y = torch.from_numpy(self.data_y)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        datax = torch.from_numpy(self.data_x[index])
        return datax, self.data_y[index]
     

class CNN256(nn.Module):
     def __init__(self):
          super(CNN256, self).__init__()
          self.conv1 = nn.Sequential(
                     nn.Conv2d(16,FILTERS, (4,1), 1, 0),
                     nn.ReLU(),#1*4*128
                     )
          self.conv2 = nn.Sequential(
                     nn.Conv2d(16, FILTERS, (1,4), 1, 0),
                     nn.ReLU(),#4*1*128
                     )
          self.conv3 = nn.Sequential(
            	     nn.Conv2d(16, FILTERS, (2,2), 1, 0),
                     nn.ReLU(), #outputsize[3*3*128]
          )
          self.conv4 = nn.Sequential(
                     nn.Conv2d(16, FILTERS, (3,3), 1, 0), 
                     nn.ReLU(),    #  2*2*128
          )
          self.conv5 = nn.Sequential(
                     nn.Conv2d(16, FILTERS, (4,4), 1, 0), #outputsize[3*3*128]
                     nn.ReLU(),
          )
          self.fc1 = nn.Linear(2816,512)
          self.fc2 = nn.Linear(512,128)
          self.fc3 = nn.Linear(128, 4)
     def forward(self, x):
         m = nn.ReLU()
         conv41 = self.conv1(x)
         conv41 = conv41.view(conv41.size(0), -1)
         conv14 = self.conv2(x)
         conv14 = conv14.view(conv14.size(0), -1)
         conv22 = self.conv3(x)
         conv22 = conv22.view(conv22.size(0), -1)

         conv33 = self.conv4(x)
         conv33 = conv33.view(conv33.size(0), -1)
         conv44 = self.conv5(x)
         conv44 = conv44.view(conv44.size(0), -1)
         hidden=torch.cat((conv41,conv14,conv22,conv33,conv44), 1)
         fc1 = self.fc1(hidden)
         fc1 = m(fc1)
         fc2 = self.fc2(fc1)
         fc2 = m(fc2)
         output = self.fc3(fc2)
         return output

cnn = CNN256()
cnn.cuda()
#cnn.load_state_dict(torch.load('256cnn_params.pkl'))

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

#for Index in range(2):
Data_Set = DealDataset()

train_loader = Data.DataLoader(dataset = Data_Set, batch_size = BATCH_SIZE, shuffle = False)
print('load complete')

for epoch in range(EPOCH):
     if epoch>=10:
         LR = 0.0001
     print('epoch{}'.format(epoch+1))
     print(time.asctime( time.localtime(time.time()) ))
     #training--------------------------
     train_loss = 0
     max_acc = 0
     train_acc = 0
     for batch_x, batch_y in train_loader:
          #print(type(batch_x))
          #print(batch_x)
          batch_x = Variable(batch_x).cuda()
          batch_y = Variable(batch_y).cuda()
          output = cnn(batch_x.float())
          #print(output)
          loss = loss_func(output,batch_y)
          train_loss+=loss.item()
          #print(train_loss)
          pred = torch.max(output,1)[1].cuda()
          train_correct = (pred == batch_y).sum()
          train_acc += train_correct.item()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
     
     print('Train Loss: {:.6f}, Acc:{:.6f}'.format(train_loss/(len(Data_Set)), train_acc/(len(Data_Set))))
     if train_acc> max_acc:
         max_acc = train_acc
         torch.save(cnn.state_dict(),"256cnn_params3.pkl")

'''
     cnn.eval()
     eval_loss = 0
     eval_acc = 0

     #evaluation--------------------------------
     for batch_x,batch_y in test_Loader:
           batch_x = Variable(batch_x)
           batch_y = Variable(batch_y)
           output = cnn(batch_x.float())
           loss = loss_func(output,batch_y)
           eval_loss+=loss.item()
           pred = torch.max(output,1)[1]
           eval_correct = (pred == batch_y).sum()
           eval_acc += eval_correct.item()
     print('Test Loss: {:.6f}, Acc:{:.6f}'.format(eval_loss/(len(test_data)), eval_acc/(len(test_data)))) 
'''
