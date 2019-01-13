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

FILTERS = 128

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
