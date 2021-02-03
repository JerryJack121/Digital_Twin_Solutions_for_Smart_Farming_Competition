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
label = np.array(label, dtype=np.float32)
print(data.shape)
print(label.shape)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Train on GPU...')
else:
    device = torch.device('cpu')

# 參數設計
batch_size = data.shape[0]
epochs = 10
train_rate = 0.7    # 訓練資料集的比例
lr = 0.0001

# 切割訓練驗證集
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
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

loss_list = []
val_loss_list = []
for epoch in range(1, epochs+1):
    print('running epoch: {} / {}'.format(epoch, epochs))
    # 訓練模式
    model.train()
    total_loss = 0
    with tqdm(trainloader) as pbar:
        for inputs, target in trainloader:
            inputs, target = inputs.to(device), target.to(device)
            inputs=inputs.permute(1,0,2)
            predict = model(inputs)
            loss = criterion(predict, target)
            running_loss = loss.item()
            total_loss += running_loss*inputs.shape[1]
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
            # 更新進度條
            pbar.set_description('train')
            pbar.set_postfix(
                    **{
                        'running_loss': running_loss,
                    })
            pbar.update(1)
    # scheduler.step()

    #評估模式
    model.eval()
    total_val_loss = 0
    with tqdm(valloader) as pbar:
        with torch.no_grad():
            for inputs, target in valloader:
                inputs, target = inputs.to(device), target.to(device)
                inputs=inputs.permute(1,0,2)
                predict = model(inputs)
                running_val_loss = criterion(predict, target).item()
                total_val_loss += running_val_loss*inputs.shape[1]
                #更新進度條
                pbar.set_description('validation')
                pbar.set_postfix(
                        **{
                            'running_val_loss': running_val_loss,
                        })
                pbar.update(1)
    loss = total_loss/len(trainloader.dataset)
    val_loss = total_val_loss/len(valloader.dataset)
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    print('train_loss: {:.4f}, valid_loss: {:.4f}, lr:{:.1e}'.format(loss, val_loss, scheduler.get_last_lr()[0]) )