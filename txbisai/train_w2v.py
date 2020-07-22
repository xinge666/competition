import os
import sys
import numpy as np 
import pandas as pd
import logging
import gc
import tqdm
import pickle
import json
import time
import tempfile
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import multiprocessing 
cwd = os.getcwd()
embed_path = os.path.join(cwd, 'embed_artifact')

# Training corpus for w2v model
corpus_dic = {
    'creative': os.path.join(embed_path, 'embed_train_creative_id_seq.pkl'),
    'ad': os.path.join(embed_path, 'embed_train_ad_id_seq.pkl'),
    'advertiser': os.path.join(embed_path, 'embed_train_advertiser_id_seq.pkl'),
    'product': os.path.join(embed_path, 'embed_train_product_id_seq.pkl'),
    'industry': os.path.join(embed_path, 'embed_train_industry_id_seq.pkl'),
    'product_category': os.path.join(embed_path, 'embed_train_product_category_id_seq.pkl')
}

def initiate_logger(log_path):
    """
    Initialize a logger with file handler and stream handler
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info('===================================')
    logger.info('Begin executing at {}'.format(time.ctime()))
    logger.info('===================================')
    return logger

def train(target, embed_size, logger=None):
    """
    Train a Word2Vec Model and save the model artifact
    """
    global corpus_dic, embed_path
    assert target in corpus_dic

    start = time.time()
    with open(corpus_dic[target], 'rb') as f:
        corpus = pickle.load(f)
    if logger: logger.info('{} corpus is loaded after {:.2f}s'.format(target.capitalize(), time.time()-start))

        
#model.build_vocab(sentences) 遍历一次语料库建立词典
#model.train(sentences) 第二次遍历语料库建立神经网络模型
 
#sg=1是skip—gram算法，对低频词敏感，默认sg=0为CBOW算法
#size是神经网络层数，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
#window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
#min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
#negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3,
#negative: 如果>0,则会采用negativesamping，用于设置多少个noise words
#hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。

    model = Word2Vec(sentences=corpus, size=embed_size, window=175, sg=1, hs=1, min_count=1, workers=multiprocessing.cpu_count())
    if logger: logger.info('{} w2v training is done after {:.2f}s'.format(target.capitalize(), time.time()-start))

    save_path = os.path.join(embed_path, '{}_sg_embed_s{}_'.format(target, embed_size))
    with tempfile.NamedTemporaryFile(prefix=save_path, delete=False) as tmp:
        tmp_file_path = tmp.name
        model.save(tmp_file_path)
    if logger: logger.info('{} w2v model is saved to {} after {:.2f}s'.format(target.capitalize(), tmp_file_path, time.time()-start))

    return tmp_file_path

def save_wv(target, embed_size, logger=None):
    global embed_path, w2v_registry
    assert target in w2v_registry

    start = time.time()
    model = Word2Vec.load(w2v_registry[target])
    save_path = os.path.join(embed_path, '{}_sg_embed_s{}_wv_'.format(target, embed_size))
    with tempfile.NamedTemporaryFile(prefix=save_path, delete=False) as tmp:
        tmp_file_path = tmp.name
        model.wv.save(tmp_file_path)
    if logger: logger.info('{} word vector is saved to {} after {:.2f}s'.format(target.capitalize(), tmp_file_path, time.time()-start))

    return tmp_file_path


if __name__=='__main__':
    assert len(sys.argv)==3
    target, embed_size = sys.argv[1], int(sys.argv[2])

    # Set up w2v model registry
    registry_path = os.path.join(embed_path, 'w2v_registry.json')
    if os.path.isfile(registry_path):
        with open(registry_path, 'r') as f:
            w2v_registry = json.load(f)
    else:
        w2v_registry = {}

    # Set up word vector registey
    wv_registry_path = os.path.join(embed_path, 'wv_registry.json')
    if os.path.isfile(wv_registry_path):
        with open(wv_registry_path, 'r') as f:
            wv_registry = json.load(f)
    else:
        wv_registry = {}

    logger = initiate_logger('train_w2v.log')

    # Train w2v model if there hasn't been one registered
    w2v_path = train(target, embed_size, logger=logger)
    w2v_registry[target] = w2v_path
    wv_path = save_wv(target, embed_size, logger=logger)
    wv_registry[target] = wv_path


    # Save word vector if there hasn't been one registered
    if target not in wv_registry:
        wv_path = save_wv(target, embed_size, logger=logger)
        wv_registry[target] = wv_path
    else:
        logger.info('{} word vector found, skip saving'.format(target.capitalize()))

    
    # Save w2v model registry
    with open(registry_path, 'w') as f:
        json.dump(w2v_registry, f)

    # Save word vector registry
    with open(wv_registry_path, 'w') as f:
        json.dump(wv_registry, f)


