!pip install torch==1.4.0
!pip install gensim
import os
import requests
cred_url = os.environ["QCLOUD_CONTAINER_INSTANCE_CREDENTIALS_URL"]
r = requests.get(cred_url)
secretId = r.json()["TmpSecretId"]
secretKey = r.json()["TmpSecretKey"]
token = r.json()["Token"]
!mkdir raw_data
!mkdir emb_layers
!mkdir prepared_data

import os
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from ti.utils import get_temporary_secret_and_token
from cos import download,upload

#### 指定本地文件路径，可根据需要修改。
local_dir = "raw_data/"
data_key="total_data.zip"
download(data_key,local_dir)

local_dir = "./"
id_word2vec="word2vec_model_128_10_1_1_0.zip"
download(id_word2vec,local_dir)

user_word2vec="word2vec_user_creative_128_1_1_0.zip"
download(user_word2vec,local_dir)

os.system("unzip raw_data/total_data.zip -d raw_data")
os.system("unzip {}".format(id_word2vec))
os.system("unzip {} -d word2vec_model".format(user_word2vec))



import pickle
import random
import pandas as pd
from multiprocessing import Pool
import random
from tqdm import tqdm

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import torch
import pickle
import gc
import torch.nn as nn
ad = pd.read_csv('raw_data/ad.csv')

def get_pretrain_embedding_layer(model_name, max_num,is_user_id=False):
    model = Word2Vec.load(model_name)
    embedding_dim = model['3'].shape[0]
    pad = np.zeros(embedding_dim)
    embeddings=[]
    for i in tqdm(range(max_num+1)):  # 从 1 开始
        try:
            if is_user_id:
                embed = model[str(-1*i)]
            else:
                embed = model[str(i)]
        except:
            embed = pad
        embeddings.append(embed)
    embeddings = np.array(embeddings)
    emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=True)
    
    if is_user_id:
        assert (emb(torch.LongTensor([max_num])).numpy()== model[str(-1*max_num)]).all()
    else:
        assert (emb(torch.LongTensor([max_num-1])).numpy()== model[str(max_num-1)]).all()
    return emb

emb = get_pretrain_embedding_layer("word2vec_model/{}.model".format(user_word2vec.split('.')[0]),max_num=4000000,is_user_id=True)
pickle.dump(emb,open('emb_layers/emb_layer_{}.pkl'.format("user_id"),'wb'),protocol=4)



del emb
gc.collect()
for col in ad.columns:
    max_num = ad[col].max()
    if col == "creative_id":
        print(col)
        emb = get_pretrain_embedding_layer("word2vec_model/{}.model".format(user_word2vec.split('.')[0]),max_num=max_num)
        pickle.dump(emb,open('emb_layers/emb_layer_{}dw.pkl'.format(col),'wb'),protocol=4)
    word2vec_model = "word2vec_model/word2vec_{}.model".format(col)
    emb = get_pretrain_embedding_layer(word2vec_model,max_num)
    pickle.dump(emb,open('emb_layers/emb_layer_{}.pkl'.format(col),'wb'),protocol=4)
    del emb
    gc.collect()


import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pickle

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
click = reduce_mem_usage(pd.read_csv('raw_data/click_log.csv'))
ad = reduce_mem_usage(pd.read_csv('raw_data/ad.csv'))
click = click.merge(ad,how='left',on='creative_id')
click.to_pickle('prepared_data/merge_data.pkl')
user_max_min_dic = {}
user_id_set = set(click.user_id.tolist())
for id_ in user_id_set:
    user_max_min_dic[id_] = {"min":float('inf'),'max':float('-inf')}
for i,id_ in enumerate(click.user_id):
    if i%10000000==0:
        print(i)
    user_max_min_dic[id_]["min"] = min(i,user_max_min_dic[id_]["min"])
    user_max_min_dic[id_]["max"] = max(i,user_max_min_dic[id_]["max"])
pickle.dump(user_max_min_dic,open('prepared_data/user_max_min_dic','wb'))
del click
del ad
del user_max_min_dic
gc.collect()