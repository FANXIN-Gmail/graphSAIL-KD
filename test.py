# -- coding:UTF-8
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [3]))

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd 

import wandb

import pdb
from collections import defaultdict
import time
import collections
# import data_utils 
# import evaluate
from shutil import copyfile

from evaluate import *
from data_utils import *

import C

wandb.login()

dataset_base_path='/data/fan_xin/Yelp'

user_num=C.user_num
item_num=C.item_num

factor_num=256
top_k=20
num_negative_test_val=-1##all

start_i_test=0
end_i_test=1000
setp=50

run_id=C.RUN_ID
print(run_id)
dataset='Yelp'

path_save_model_base='/data/fan_xin/newlossModel_graphSAIL/'+dataset+'/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    pdb.set_trace()

training_user_set, training_item_set = json_to_set(dataset_base_path + "/check_in.json", [0.6, 0.7])
testing_user_set = json_to_set(dataset_base_path + "/check_in.json", [0.7, 0.75], single=1)

training_user_set[user_num-1].add(item_num-1)
training_item_set[item_num-1].add(user_num-1)

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)

model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u)
model = model.to('cuda')

run = wandb.init(
    # Set the project where this run will be logged
    project="KD-graphSail-Yelp",
    # notes="random_without_remap",
    # tags=["ramdom", "10%"],
    name=run_id,
    mode="online",
)

########################### TRAINING ##################################### 
# testing_loader_loss.dataset.ng_sample() 

print('--------test processing-------') 
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):

    test_start_time = time.time()

    model.train()
    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), PATH_model)
    model.load_state_dict(torch.load(PATH_model))

    model.eval()
    # ######test and val###########  

    with torch.no_grad():

        gcn_users_embedding, gcn_items_embedding = model.inference() 

        user_e=gcn_users_embedding.cpu().detach().numpy()
        item_e=gcn_items_embedding.cpu().detach().numpy()

        all_pre=np.matmul(user_e,item_e.T)

        # all_pre=torch.matmul(gcn_users_embedding, gcn_items_embedding.T)

        # all_pre=all_pre.cpu().numpy()

    NDCG, PRECISION, RECALL = [], [], []
    set_all=set(range(item_num)) 
    #spend 461s
    # test_start_time = time.time()
    for u_i in testing_user_set:

        item_i_list = list(testing_user_set[u_i])
        index_end_i = len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list)

        pre_one=all_pre[u_i][item_i_list]

        indices=largest_indices(pre_one, top_k)
        indices=list(indices[0])

        recall_t,precision_t,ndcg_t=hr_ndcg(indices,index_end_i,top_k)

        # elapsed_time = time.time() - test_start_time

        NDCG.append(ndcg_t)
        PRECISION.append(precision_t)
        RECALL.append(recall_t)

    ndcg_test = round(np.mean(NDCG),4)
    precision_test = round(np.mean(PRECISION),4)
    recall_test = round(np.mean(RECALL),4)

    elapsed_time = time.time() - test_start_time

    # test_loss,hr_test,ndcg_test = evaluate.metrics(model,testing_loader,top_k,num_negative_test_val,batch_size)  
    str_print_evl="epoch:"+str(epoch)+' time:'+str(round(elapsed_time, 2))+"\t test"+ ' ndcg:'+str(ndcg_test)+ " recall:"+str(recall_test)+ " precision:"+str(precision_test)

    wandb.log({"recall":recall_test})

    print(str_print_evl)

