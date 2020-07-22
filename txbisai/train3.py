import os
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from collections import  OrderedDict
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import pickle
from torch.utils.data import DataLoader
import random
from utils import save_model,test_model,PositoinEncoder,TransformerOptimizer,logger
from dataset import MyDataSet
merge_data = pd.read_pickle('prepared_data/merge_data.pkl')
user_max_min_dic = pickle.load(open("prepared_data/user_max_min_dic",'rb'))
user_df = pd.read_csv('prepared_data/user.csv')

use_col = [  'creative_id',  'ad_id', 'product_id',
       'product_category', 'advertiser_id', 'industry']

#加载预处理的embedding layer

emblayer_dic = {}
emblayer_dic["user_id"] = pickle.load(open("emb_layer/emb_layer_user_id.pkl",'rb'))

for col in use_col:
    emblayer_dic[col] = pickle.load(open("emb_layer/emb_layer_{}.pkl".format(col),'rb'))

class MyModel(nn.Module):
    def __init__(self,modeltype,use_col,dmodel = 512,):
        super(MyModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(dmodel,nhead=8)
        encoder_norm = nn.LayerNorm(dmodel)
        self.encoder = nn.TransformerEncoder(encoder_layer, 1, encoder_norm)

        self.ufc = nn.Linear(256,dmodel)
        #self.user_creative_emb_layer = [emblayer_dic['user_id'].cuda(),None]#emblayer_dic['creative_id_dw'].cuda()]
        #self.emb_layers = [emblayer_dic[key].cuda() for key in use_col]
        self.char_embedding_size = 0
        self.dropout = nn.Dropout(0.2)
        self.char_embedding_size = 16
        self.emb_lay =  nn.Embedding(92,16)
        self.cfc = nn.Linear(400*6+1 + 8 + 16,dmodel)
        self.rnn = nn.LSTM(dmodel, 256, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=0.2,
                            bidirectional=True)
        self.convs = nn.ModuleList([nn.Conv2d(1,128,(K,512)) for K in [2,4,8]])
        self.outfc1 = nn.Linear(128*3,64)
        self.outfc2 = nn.Linear(64,10)
        self.outfc3 = nn.Linear(64,2)
        #self.vgg_block2 = vgg_block(3,64,128)
    def get_dim(self,dim):
        return int(12*(1+np.log(dim**0.5)))
    
    def vgg_block(self,num_convs, in_channels, out_channels):
        # 定义第一层，并转化为 List
        net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),nn.ReLU(True)]

        # 通过循环定义其他层
        for i in range(num_convs - 1):
            net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1))
            net.append(nn.ReLU(True))
        net.append(nn.MaxPool2d(2, 2))
        # List数据前面加‘*’表示将List拆分为独立的参数
        return nn.Sequential(*net)


    def forward(self,user_id_emb,raw_arr,click,d_time):
        self.rnn.flatten_parameters()
        size = d_time.size()
        d_time = self.emb_lay(d_time.long().view(size[1]*size[0])).view(size[0],size[1],-1)
        cid = torch.cat([raw_arr,click.unsqueeze(2),d_time],dim=2)
        cid = self.cfc(cid)
        cid = self.dropout(cid)
        cid = self.encoder(cid)
        x = self.rnn(cid)[0]
        x = x.unsqueeze(1)
        x = self.dropout(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        out = self.outfc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out1 = self.outfc2(out)
        out2 = self.outfc3(out)
        return out1,out2

def train(continue_from,train_loader,test_loader):
    print(continue_from)
    #continue_from = "best_transformer_age5.pth"
    modeltype = "age"
    model = MyModel(modeltype,use_col)
    model = model.cuda()

    #warm up optimizer
    optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-08),
            0.8,
            512,
            8000)

    start_epoch = 0
    print("start load")
    if os.path.exists(continue_from ):
            naive_model = torch.load(continue_from , map_location=lambda storage, loc:storage)
            new_state_dict = OrderedDict()
            for k, v in naive_model["state"].items():
                    new_state_dict[k] = v 
            model.load_state_dict(new_state_dict)
            start_epoch = naive_model["epoch"]+1
            optimizer.load_state_dict(naive_model["optimizer"])
            optimizer.step_num = naive_model['optimizer_step']
            logger.info("load model finish")
            print("end load")
    losses_age = []
    losses_gender= []
    best_acc_age = 0 
    best_acc_gender = 0 
    entroy=nn.CrossEntropyLoss()#mseloss = nn.MSELoss()
    model = nn.DataParallel(model)
    best_age_acc = 0
    best_gender_acc = 0
    #acc_age,acc_gender,loss_age,loss_gender = test_model(model,test_loader,entroy)
    for epo in range(start_epoch,30):
        model.train()
        #start = time.time()
        for i,(data) in enumerate(train_loader):
            raw_arr,click,times,label_age,label_gender,user_id,user_id_emb = data
            label_age = label_age.cuda()
            label_gender = label_gender.cuda()
            click = click.cuda()
            times= times.cuda()
            user_id_emb = user_id_emb.cuda()
            raw_arr= raw_arr.cuda()
            embs1,embs2  = model(user_id_emb,raw_arr,click,times)
            loss_age = entroy(embs1,label_age-1)
            loss_gender = entroy(embs2,label_gender-1)
            loss = loss_age+loss_gender
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
            optimizer.step()
            losses_age.append(loss_age.item())
            losses_gender.append(loss_gender.item())
            if (i+1)%200==0: 
                #print((time.time()-start)/200*len(train_loader))
                for g in optimizer.optimizer.param_groups:
                    logger.info("now learn rate : "+str(g['lr']) )
                mean_loss_age = sum(losses_age)/len(losses_age)
                mean_loss_gender = sum(losses_gender)/len(losses_gender)
                logger.info('cpEpoch:{0} \t step:{1}/{2} \t '
                                    'mean loss:{3} \t{4}'.format(  
                                                            (epo + 1), (i + 1),len(train_loader),mean_loss_age,mean_loss_gender))
        acc_age,acc_gender,loss_age,loss_gender = test_model(model,test_loader,entroy)
        logger.info("acc:{},{} loss:{}{}".format(acc_age,acc_gender,loss_age,loss_gender))
        if acc_age>best_acc_age:
            best_acc_age = acc_age
            save_model(model,epo,optimizer,model_name = continue_from)
        if acc_gender>best_gender_acc:
            best_gender_acc = acc_gender
            save_model(model,epo,optimizer,model_name = continue_from + "gender.pth")


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
modeltype = 'age'
batch_size = 400
skf = StratifiedKFold(n_splits=5, random_state=10086, shuffle=True) 
userid_ls =np.array([i+1 for i in range(3000000)])
id_ls =np.array([0 for i in range(3000000)])
for index, (train_index, test_index) in enumerate(skf.split(userid_ls.reshape(-1,1),id_ls)):
    if index <4:
        continue
    flod_dic = pickle.load(open('flod_dic.pkl','rb'))
    if not index ==4:
        continue
    torch.cuda.empty_cache()
    train_x, test_x = flod_dic[index]
    #train_x, test_x = userid_ls[train_index,], userid_ls[test_index,]
    continue_fom = "age_tf_pre_encoder_8_5fload_{}.pth".format(index)
    train_dst = MyDataSet(train_x,merge_data ,user_df ,user_max_min_dic,emblayer_dic = emblayer_dic,user_columns =use_col )
    train_loader = DataLoader(train_dst,num_workers=6,batch_size=600,shuffle=True)
    test_dst = MyDataSet(test_x,merge_data ,user_df ,user_max_min_dic,user_columns =use_col ,emblayer_dic = emblayer_dic,is_train=False)
    test_loader = DataLoader(test_dst,num_workers=10,batch_size=600)
    #break
    train(continue_fom,train_loader,test_loader)

