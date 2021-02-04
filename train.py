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
from sklearn.metrics import f1_score, hamming_loss

PATH = r'D:\dataset\2021智慧農業數位分身創新應用競賽\generate_dateset'

# 載入訓練資料
data = np.load(os.path.join(PATH, 'train-val_data.npy'), allow_pickle=True)
label = np.load(os.path.join(PATH, 'train-val_label.npy'), allow_pickle=True)
label = np.array(label, dtype=np.float32)
print(data.shape)
print(label.shape)

# 打散資料
indices = np.random.permutation(data.shape[0])
data = data[indices]

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Train on GPU...')
else:
    device = torch.device('cpu')

# 參數設計
# batch_size = data.shape[0]
batch_size = 15000
epochs = 100
train_rate = 0.8    # 訓練資料集的比例
lr = 1e-3
threshold = torch.tensor([0.5])

# 切割訓練驗證集
train_num = int(data.shape[0]*train_rate)
train_x = torch.tensor(data[:train_num], dtype=torch.float)
train_y = torch.tensor(label[:train_num])
val_x = torch.tensor(data[train_num:], dtype=torch.float)
val_y = torch.tensor(label[train_num:])
trainset = Setloader(train_x, train_y)
valset = Setloader(val_x, val_y)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

# 定義模型
model = rnn.rnn(input_size=18, output_size=11)
model.to(device)

# 載入預訓練權重
# model.load_state_dict(torch.load('./weights/epoch100-loss0.0736-val_loss0.0677-f10.7676.pth'))

# 定義優化器、損失函數
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
#                                            T_max=20,
#                                            eta_min=1e-6,
#                                            last_epoch=-1)

loss_list = []
val_loss_list = []
f1_list = []
lr_list = []
for epoch in range(1, epochs+1):
    print('\nrunning epoch: {} / {}'.format(epoch, epochs))
    # 訓練模式
    model.train()
    total_loss = 0
    with tqdm(trainloader) as pbar:
        for inputs, target in trainloader:
            inputs, target = inputs.to(device), target.to(device)
            inputs=inputs.permute(1,0,2)    # (sequence, batch, data)
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
                        'lr': optimizer.state_dict()['param_groups'][0]['lr']
                    })
            pbar.update(1)
    scheduler.step()

    #評估模式
    model.eval()
    outputs_list = np.empty((0,11))
    total_val_loss = 0
    with tqdm(valloader) as pbar:
        with torch.no_grad():
            for inputs, target in valloader:
                inputs, target = inputs.to(device), target.to(device)
                inputs=inputs.permute(1,0,2)
                outputs = model(inputs)
                running_val_loss = criterion(outputs, target).item()
                total_val_loss += running_val_loss*inputs.shape[1]
                outputs = (outputs.cpu() > threshold).float()*1
                outputs_list = np.vstack((outputs_list, outputs))
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
    f1 = f1_score(val_y, outputs_list, average="macro")
    f1_list.append(f1)
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    print('train_loss: {:.4f}, valid_loss: {:.4f}'.format(loss, val_loss))
    print('f1_score:{:.6f}, hamming_loss:{:.4f}'.format(f1, hamming_loss(val_y, outputs_list)))
    torch.save(model.state_dict(), './logs/epoch%d-loss%.4f-val_loss%.4f-f1%.4f.pth' %(epoch, loss, val_loss, f1))

#繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss_list, label='Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend(loc='best')
plt.savefig('./images/loss.jpg')
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('f1 score')
plt.plot(f1_list)
plt.savefig('./images/f1.jpg')
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.plot(lr_list, label='lr')
plt.savefig('./images/lr.jpg')
plt.show()