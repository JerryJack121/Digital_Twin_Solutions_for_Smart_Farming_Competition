import numpy as np 
import os
import pandas as pd
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import rnn
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from utils.Setloader import Setloader

PATH = r'D:\dataset\2021智慧農業數位分身創新應用競賽\generate_dateset'

# 載入訓練資料
data = np.load(os.path.join(PATH, 'train-val_data.npy'), allow_pickle=True)
label = np.load(os.path.join(PATH, 'train-val_label.npy'), allow_pickle=True)
label = np.array(label, dtype=np.float16)
print(data.shape)
print(label.shape)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Train on GPU...')
else:
    device = torch.device('cpu')

# 參數設計
batch_size = 3
epochs = 10
train_rate = 0.7

#正規化
x_scaler = StandardScaler().fit(data)
data = x_scaler.transform(data)

train_num = int(data.shape[0]*train_rate)
train_x = torch.tensor(data[:train_num], dtype=torch.float)
train_y = torch.tensor(label[:train_num])
val_x = torch.tensor(data[:train_num], dtype=torch.float)
val_y = torch.tensor(label[:train_num])
trainset = Setloader(train_x, train_y)
valset = Setloader(val_x, val_y)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

# 定義模型
model = rnn.rnn(input_size=18, output_size=11)
model.to(device)

# 定義優化器、損失函數
criterion = nn.MSELoss().to(device)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# 訓練模式
model.train()
for inputs, target in trainloader:
    inputs, target = inputs.to(device), target.to(device)
    inputs=inputs.permute(1,0,2)
    predict = model(inputs)
    loss = criterion(predict, target)
    running_loss = loss.item()
    print(running_loss)



