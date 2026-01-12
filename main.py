import os
import torch
import math
from exp.exp_main import exp_main
import numpy as np
import random

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_battery = '18'
start = 20
task = 'SOH'
class Args:
    def __init__(self):
        self.model = 'SAR_net'
        # self.model = 'Transformer'
        self.soh_att = 'PSA'
        self.rul_att = 'SPA'
        self.model_id = 'LA{}#{}_{}'.format(start,train_battery,task)
        self.task = 'SOH'
        self.battery = 'NASA'
        self.results_path = './results/NASA/{}/'.format(self.task)
        self.checkpoints = './checkpoints/'
        self.root_path = './datasets/NASA/'
        self.train_battery = train_battery
        self.train_battery_now = train_battery
        self.data_path = 'battery_data_frames[].csv'
        self.start = start
        self.norm = False

        self.epochs = 40
        self.patience = 15
        self.optim = 'adam'

        self.warmup_epochs = 10
        self.min_lr = 0
        self.smoothing_learning_rate = 0
        self.damping_learning_rate = 0
        self.lradj = 'exponential_with_warmup'

        self.learning_rate = 0.0001  #0.0001
        self.batch_size = 16 #16
        self.d_ff = 32  #32
        self.d_mlp = 128
        self.d_xlstm = 256
        self.f_xlstm = 1/4
        self.d_model = 64  #64
        self.n_heads = 4
        self.e_layers = 1
        self.d_layers = 1
        self.dropout = 0.1
        self.xlstm_layer = ['m','m']

        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 1
        self.enc_in = 1
        self.dec_in = 1
        self.seq_len = 3997
        self.label_len = self.seq_len
        self.c_out = 1

        self.grad_clip = 1.0  #
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

args = Args()
exp = exp_main(args)
seed_everything(11)
print('<<<<<<<<<<<<<<<<<<<<<<<<< start training >>>>>>>>>>>>>>>>>>>>>>>>>')
# exp.train()
exp.test(Time_record=False)

torch.cuda.empty_cache()