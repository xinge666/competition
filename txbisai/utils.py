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
import logging
logger = logging.getLogger(__name__)                                                                                                                                                                                                          
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('age_lstm_dfm2.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

def save_model(net,epoch,optimizer,model_name = 'best_transformer.pth'):
    if isinstance(net,torch.nn.DataParallel):
        net = net.module
    state = { 
                "epoch": epoch,
                "state": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_step":optimizer.step_num
            }   
    torch.save(state, model_name)
def test_model(model,test_loader,entroy):
    model = model.eval()
    predict_age = []
    predict_gender = []
    label_ages = []
    label_genders = []
    losses_age = []
    losses_gender= []
    total_len = len(test_loader)
    with torch.no_grad():
        for i,(data) in enumerate(test_loader):
            if i%(int(total_len/5))==0:
                print(i,len(test_loader))
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
            losses_age.append(loss_age.item())
            losses_gender.append(loss_gender.item())
            predict_age.extend(torch.max(embs1.cpu(),1)[1].numpy()+1)
            predict_gender.extend(torch.max(embs2.cpu(),1)[1].numpy()+1)
            label_ages.extend(label_age.cpu().numpy())
            label_genders.extend(label_gender.cpu().numpy())
            acc_age = accuracy_score(label_ages,np.floor(predict_age))
            acc_gender = accuracy_score(label_genders,np.floor(predict_gender))
    model.train()
    return acc_age,acc_gender,sum(losses_age)/len(losses_age),sum(losses_gender)/len(losses_gender)

class PositoinEncoder(nn.Module):
    def __init__(self,max_len=100,dmodel=512):
        super(PositoinEncoder, self).__init__()
        pe = torch.zeros(max_len, dmodel, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() *
                             -(math.log(10000.0) / dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda()
        
    def forward(self,x):
        size = x.size()
        return self.pe[x.view(size[0]*size[1]),:].view(size[0],size[1],-1).transpose(0,1)


class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.visdom_lr = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self._visdom()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_visdom(self, visdom_lr, vis):
        self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
        self.vis = vis  # visdom enviroment
        self.vis_opts = dict(title='Learning Rate',
                             ylabel='Leanring Rate', xlabel='step')
        self.vis_window = None
        self.x_axis = torch.LongTensor()
        self.y_axis = torch.FloatTensor()

    def _visdom(self):
        if self.visdom_lr is not None:
            self.x_axis = torch.cat(
                [self.x_axis, torch.LongTensor([self.step_num])])
            self.y_axis = torch.cat(
                [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
            if self.vis_window is None:
                self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
                                                opts=self.vis_opts)
            else:
                self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
                              update='replace')