import os
import numpy as np
import glob
import random
from tqdm import tqdm
import datetime

import torch
from torch.utils import data
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from visualize import *
from crowdnet_pytorch import CrowdNet
from hdf5_dataset import HDF5Dataset

model_name = 'dcc_crowdnet'
model_path = os.path.expanduser(os.path.join('./models', model_name))
data_path = os.path.expanduser(os.path.join('./data', model_name))

dataset_paths = 'dataset/UCF_CC_50'

model = CrowdNet(True)
if torch.cuda.is_available():
    model.cuda()
    
model_name = 'xx'
model.load_state_dict(torch.load(os.path.join(model_path,'cc_epoch_{}.mdl'.format(model_name))))

print(model)

hdf5_train_ds = HDF5Dataset(os.path.join(data_path,'train.txt'))
hdf5_train_loader = data.DataLoader(dataset=hdf5_train_ds, batch_size=50, shuffle=False)

hdf5_test_ds = HDF5Dataset(os.path.join(data_path,'test.txt'))
hdf5_test_loader = data.DataLoader(dataset=hdf5_test_ds, batch_size=50, shuffle=False)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
metric = nn.L1Loss()
        
# Train Model
pbar_train = tqdm(enumerate(hdf5_train_loader), total=len(hdf5_train_loader))
total_mae = 0
for i, data in pbar_train:
    src_img = Variable(torch.Tensor(data[0]))
    trg_img = Variable(torch.Tensor(data[1]))

    if torch.cuda.is_available():
        src_img = src_img.cuda()
        trg_img = trg_img.cuda()

    predict_img = model(src_img)
    loss = metric(predict_img, trg_img)
    total_mae += loss.data[0]
        
    pbar_train.set_description('MAE:{:.4f}'.format(total_mae))
print("MAE Metric : {}".format(total_mae))

# Evaluate Model
pbar_test = tqdm(enumerate(hdf5_test_loader), total=len(hdf5_test_loader))
for j, data in pbar_test:
    src_img = Variable(torch.Tensor(data[0]))
    trg_img = Variable(torch.Tensor(data[1]))

    if torch.cuda.is_available():
        src_img = src_img.cuda()
        trg_img = trg_img.cuda()

    predict_img = model(src_img)
    loss = metric(predict_img, trg_img)
    total_mae += loss.data[0]

    pbar_test.set_description("MAE:{:.4f}".format(total_mae))
print("MAE Metric : {}".format(total_mae))
