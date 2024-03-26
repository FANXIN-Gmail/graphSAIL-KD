# -- coding:UTF-8
import numpy as np
import torch 
import time
import pdb
import math
import collections
import json

# Read json to list
def read(url, rate=[0, 1]):

    All = list()

    with open(url, mode='r') as f:

        line = f.readline()

        while line:
        
            line = json.loads(line)
            
            All.append(line)
            
            line = f.readline()

    f.close()

    return All[round(len(All)*rate[0]):round(len(All)*rate[1])]

# Generate Set from List
def list_to_set(All, rate=[0, 1], single=0):

    training_user_set = collections.defaultdict(set)
    training_item_set = collections.defaultdict(set)

    for line in All[round(len(All)*rate[0]):round(len(All)*rate[1])]:
        training_user_set[line["user_id"]].add(line["item_id"])
        training_item_set[line["item_id"]].add(line["user_id"])

    if single:
        return training_user_set
    else:
        return training_user_set, training_item_set


# Generate trainSet from json
def json_to_set(url, rate=[0, 1], single=0):

    training_user_set = collections.defaultdict(set)
    training_item_set = collections.defaultdict(set)

    All = list()
    
    with open(url, mode='r') as f:

        line = f.readline()

        while line:

            line = json.loads(line)

            All.append(line)
    
            line = f.readline()

    f.close()

    for line in All[round(len(All)*rate[0]):round(len(All)*rate[1])]:
        training_user_set[line['user_id']].add(line['item_id'])
        training_item_set[line['item_id']].add(line['user_id'])

    if single:
        return training_user_set
    else:
        return training_user_set, training_item_set

# Count the number of interations
def count_interaction(Set):

    count = 0
    for k,v in Set.items():
        for i in v:
            count += 1
    return count

def count_user_item(Set):

    I = set()
    for k,v in Set.items():
        for i in v:
            I.add(i)
    return len(Set), len(I)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break

    if max_hr == 0:
      hr_t = 0
    else:
      hr_t=hr_topK/index_end_i
      pre_t = hr_topK/top_k
    
    ndcg_t=ndcg_topK/max_ndcg  
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace() 
    return hr_t,pre_t,ndcg_t


def metrics_loss(model, test_val_loader_loss, batch_size): 
    start_time = time.time() 
    loss_sum=[]
    loss_sum2=[]
    for user, item_i, item_j in test_val_loader_loss:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() 
     
        prediction_i, prediction_j,loss,loss2 = model(user, item_i, item_j) 
        loss_sum.append(loss.item())  
        loss_sum2.append(loss2.item())

        # if np.isnan(loss2.item()).any():
        #     pdb.set_trace()
    # pdb.set_trace()
    elapsed_time = time.time() - start_time
    test_val_loss1=round(np.mean(loss_sum),4)
    test_val_loss=round(np.mean(loss_sum2),4)#round(np.mean(loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_val_loss=' val loss:'+str(test_val_loss)#+' time:'+str(round(elapsed_time,3))+' s'
    # print(round(elapsed_time,3))
    # print(test_val_loss1,test_val_loss)
    return test_val_loss

def metrics(model, test_val_loader, top_k, num_negative_test_val, batch_size):
    HR, NDCG = [], [] 
    test_loss_sum=[]
    # pdb.set_trace()  
 
    test_start_time = time.time()
    for user, item_i, item_j in test_val_loader:  
        # start_time = time.time()
        # pdb.set_trace()
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j #index to split

        prediction_i, prediction_j,loss_test,loss2_test = model(user, item_i, torch.cuda.LongTensor([0])) 
        test_loss_sum.append(loss2_test.item())  
        # pdb.set_trace()   
        elapsed_time = time.time() - test_start_time
        print('time:'+str(round(elapsed_time,2)))
        courrent_index=0
        courrent_user_index=0
        for len_i,len_j in item_j:
            index_end_i=(len_i-len_j).item()  
            #pre_error=(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)])#.sum() 
            #loss_test=nn.MSELoss((pre_error).sum())#-(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)]).sigmoid().log()#.sum()   
            _, indices = torch.topk(prediction_i[0][courrent_index:(courrent_index+len_i)], top_k)   
            hr_t,ndcg_t=hr_ndcg(indices.tolist(),index_end_i,top_k)  
            # print(hr_t,ndcg_t,indices,index_end_i)
            # pdb.set_trace()
            HR.append(hr_t)
            NDCG.append(ndcg_t) 

            courrent_index+=len_i 
            courrent_user_index+=1 

 
    test_loss=round(np.mean(test_loss_sum[:-1]),4)  
 
    return test_loss,round(np.mean(HR),4) , round(np.mean(NDCG),4) 


# export ALL_PROXY=http://www-proxy.waseda.jp:8080/
# export http_proxy=$ALL_PROXY
# export https_proxy=$ALL_PROXY

# ./configure --prefix=${HOMW}/.local
# make
# make install


