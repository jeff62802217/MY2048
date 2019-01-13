from agents import Agent
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
from cnn_model import CNN256

map_table = {2**i: i for i in range(1, 16)}
map_table[0] = 0
def grid_arr(arr):
    ret = np.zeros(shape=(16,4, 4),dtype = float)
    for r in range(4):
        for c in range(4):
            ret[map_table[arr[r][c]]][r][c] = 1
    return ret

class MyAgent(Agent):
    def __init__(self,game, display =None):
        self.model = CNN256()
        self.model.load_state_dict(torch.load('256cnn_params3.pkl'))
        super().__init__(game, display)
        
    def step(self): 
        Input = np.expand_dims(grid_arr(self.game.board) , axis=0)
        output = self.model.forward(torch.from_numpy(Input).float())
        direction = torch.max(output,1)[1]
        return direction
   
        
