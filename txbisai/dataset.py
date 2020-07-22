import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import pickle
from torch.utils.data import DataLoader
import random
def time2week(a):
    return a%7

def time2holiday(a):
    a=a+6#time=1是周日，7
    if(a%7==0):
        b=7
    else:
        b= (a%7)
    if(b==6 or b==7 or (a >=37 and a<=43)):#31~37 +6
        return 1
    else:
        return 0
    
class MyDataSet(Dataset):
    def __init__(self,user_id_ls,
                 merge_data = None,
                 user_df=None ,
                 user_max_min_dic = None,
                 user_columns = None,
                 emblayer_dic = None,
                 is_train = True,
                 is_eval = False,
                 sample_rate=0.6,
                 data_dir = './',
                 shuffle=False):
        self.is_eval = is_eval
        self.is_train = is_train
        self.user_max_min_dic=user_max_min_dic
        self.user_df = user_df
        self.user_columns = user_columns
        self.merge_data = merge_data
        self.user_df = user_df
        self.emblayer_dic = emblayer_dic
        self.user_id_ls = user_id_ls 
        self.sample_rate =sample_rate
        self.data_dir = data_dir
        self.shuffle = shuffle
    def pad_to_longest(self,insts, max_len=100):
        if insts.shape[0]<max_len:
            return np.pad(insts,((0,max_len-insts.shape[0]),(0,0)))
        return insts[:100,:]
    def __len__(self):
        return len(self.user_id_ls)
    def __getitem__(self, index):
        user_id = self.user_id_ls[index]
        temp_df = self.merge_data.iloc[self.user_max_min_dic[user_id]["min"]:self.user_max_min_dic[user_id]["max"]+1,:]
        raw_arr = temp_df[self.user_columns ]
        click_and_time = temp_df[['click_times','time']].values
        click = temp_df['click_times'].values
        time = temp_df['time'].values
        time = temp_df['time'].values
        time0 = temp_df['time'].apply(time2week).values
        time1 = temp_df['time'].apply(time2holiday).values
        raw_arr = raw_arr.values
        sample_rate = random.uniform(self.sample_rate,1)
        length = raw_arr.shape[0]
        sample_arr = np.random.choice([i for i in range(length)],int(np.ceil(length*sample_rate)),replace=False)
        if not self.shuffle:
            sample_arr.sort()
        if self.is_train:
            click = click[sample_arr]
            time = time[sample_arr]
            raw_arr = raw_arr[sample_arr,:]
        raw_arr = torch.LongTensor(raw_arr)
        #return raw_arr
        emb_ls = []
        for i,col in enumerate(self.user_columns):
            emb_ls.append(self.emblayer_dic[col](raw_arr[:,i]))
        #return emb_ls
        raw_arr = torch.cat(emb_ls,dim=1)
        user_id_emb = self.emblayer_dic['user_id'](torch.LongTensor([user_id]))
        time = torch.nn.functional.pad(torch.Tensor(time),(0,100-time.shape[0]))
        time0 = torch.nn.functional.pad(torch.Tensor(time0),(0,100-time0.shape[0]))
        time1 = torch.nn.functional.pad(torch.Tensor(time1),(0,100-time1.shape[0]))
        click = torch.nn.functional.pad(torch.Tensor(click),(0,100-click.shape[0]))
        raw_arr = torch.nn.functional.pad(raw_arr,(0,0,0,100-raw_arr.shape[0]))
        time0 = torch.zeros(100, 7).scatter_(1, time0.view(100,-1).long(),1)
        raw_arr = torch.cat([raw_arr,time0,time1.view(100,-1)],dim = 1)
        if self.is_eval:
            return raw_arr,click,time,user_id,user_id_emb[0]
        else:
            label_gender = self.user_df.gender[user_id-1]
            label_age = self.user_df.age[user_id-1]
            return raw_arr,click,time,label_age,label_gender,user_id,user_id_emb[0]