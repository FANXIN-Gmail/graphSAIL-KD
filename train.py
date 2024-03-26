# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import wandb

import argparse
import os
import numpy as np
import math
import sys

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

os.environ["WANDB_API_KEY"] = "15b5e8572b0516899bae70d5bbb5c9091d1667a7"

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.cluster import KMeans

import pdb
from collections import defaultdict
import time
import collections
from shutil import copyfile

from evaluate import *
from data_utils import *
import C

wandb.login()

dataset_base_path='/data/fan_xin/Yelp'

epoch_num=C.EPOCH

user_num=C.user_num
item_num=C.item_num

factor_num=256
batch_size=1024*4
learning_rate=C.LR

num_negative_test_val=-1##all

run_id=C.RUN_ID
print(run_id)
dataset='Yelp'

path_save_model_base='/data/fan_xin/newlossModel_graphSAIL/'+dataset+'/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

base = read(dataset_base_path + "/check_in.json", [0, 0.6])
block = read(dataset_base_path + "/check_in.json", [0.6, 0.7])
training_user_set, training_item_set = list_to_set(block)
training_user_set_, training_item_set_ = list_to_set(base)
training_set_count = count_interaction(training_user_set)
user_rating_set_all = json_to_set(dataset_base_path + "/check_in.json", single=1)

print(training_set_count)

training_user_set[user_num-1].add(item_num-1)
training_item_set[item_num-1].add(user_num-1)
training_user_set_[user_num-1].add(item_num-1)
training_item_set_[item_num-1].add(user_num-1)

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
u_d_=readD(training_user_set_,user_num)
i_d_=readD(training_item_set_,item_num)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)
sparse_u_i_=readTrainSparseMatrix(training_user_set_,u_d_,i_d_,True)
sparse_i_u_=readTrainSparseMatrix(training_item_set_,u_d_,i_d_,False)

train_dataset = BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count, all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)

new_U_set, new_I_set = training_user_set, training_item_set
old_U_set, old_I_set = training_user_set_, training_item_set_

old_sparse_u_i=readTrainSparseMatrix(old_U_set,u_d_,i_d_,True,False)
old_sparse_i_u=readTrainSparseMatrix(old_I_set,u_d_,i_d_,False,False)

PATH_model='/data/fan_xin/newlossModel_graphSAIL/'+dataset+'/'+C.BASE+'/epoch'+str(C.BASE_EPOCH)+'.pt'

model_ = BPR(user_num, item_num, factor_num, sparse_u_i_, sparse_i_u_).to('cuda')
model_.load_state_dict(torch.load(PATH_model))
model_.eval()
with torch.no_grad():
    old_U_emb, old_I_emb = model_.inference() 

model = BPR(user_num,item_num,factor_num,sparse_u_i,sparse_i_u,
    old_sparse_u_i=old_sparse_u_i,
    old_sparse_i_u=old_sparse_i_u,
    old_U_emb=old_U_emb,
    old_I_emb=old_I_emb,
    ).to('cuda')
model.load_state_dict(torch.load(PATH_model))
model.pre_self(new_U_set, new_I_set, old_U_set, old_I_set)

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.5, 0.99))

run = wandb.init(
    # Set the project where this run will be logged
    project="KD-graphSail-Yelp",
    # notes="random_without_remap",
    # tags=["ramdom", "10%"],
    name=run_id,
    mode="offline",
)

########################### TRAINING #####################################

# testing_loader_loss.dataset.ng_sample()

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(epoch_num):

    model.train() 
    start_time = time.time()

    train_loader.dataset.ng_sample()

    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()

    train_loss_sum=[]
    loss_BPR_sum=[]
    loss_self_sum=[]
    loss_local_sum=[]
    loss_global_sum=[]

    for user, item_i, item_j in train_loader:

        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        model.zero_grad()
        loss = model(user, item_i, item_j)
        sum(loss).backward()
        optimizer_bpr.step()
        count += 1
        train_loss_sum.append(sum(loss).item())  
        loss_BPR_sum.append(loss[0].item())
        loss_self_sum.append(loss[1].item())
        loss_local_sum.append(loss[2].item())
        loss_global_sum.append(loss[3].item())

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)
    loss_BPR=round(np.mean(loss_BPR_sum[:-1]),4)
    loss_self=round(np.mean(loss_self_sum[:-1]),4)
    loss_local=round(np.mean(loss_local_sum[:-1]),4)
    loss_global=round(np.mean(loss_global_sum[:-1]),4)

    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)
    # print('--train--',elapsed_time)

    wandb.log({"train_loss": train_loss, 
        "loss_BPR": loss_BPR, 
        "loss_self": loss_self,
        "loss_local": loss_local,
        "loss_global": loss_global})

    print(str_print_train)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)
