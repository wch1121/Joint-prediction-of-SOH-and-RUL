import torch
import math
import torch.nn as nn
from Attention.CBAM import CBAM
from Attention.ACmixAttention import ACmix1D
from Attention.S2Attention import S2Attention
from Attention.TripletAttention import TripletAttention
from Attention.EMSA import EMSA
from Attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention
from Attention.SGE import SpatialGroupEnhance
from Attention.SKAttention import SKAttention
from Attention.SparseSelfAttention import SparseSelfAttention
from Attention.ProbAttention import Prob_AttentionLayer
from models.BZY.xLSTM import xLSTM

class SAR_N(nn.Module):
    def __init__(self, configs):
        super(SAR_N, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.configs = configs
        self.task = configs.task
        self.Backbone = InceptionModule(configs)
        # self.conv = nn.Conv1d(in_channels=configs.c_out,out_channels=configs.d_model,kernel_size=3,padding=1)
        self.branch1 = nn.Sequential(
            nn.Conv1d(configs.c_out, configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.Linear = nn.Linear(configs.c_out, configs.d_model)
        self.GRU = GRU(configs)
        self.xlstm = xLSTM(configs)
        layers_soh = []
        layers_rul = []
        for i in range(configs.e_layers):
            layers_soh += [SOHBlock(configs=configs)]
        self.SOH_network = nn.Sequential(*layers_soh)
        for i in range(configs.e_layers):
            layers_rul += [RULBlock(configs=configs)]
        self.RUL_network = nn.Sequential(*layers_rul)

        SOH_output_size = self.configs.d_model * self.configs.seq_len   # This needs to be calculated based on conv/pool operations
        RUL_output_size = self.configs.d_model * self.configs.seq_len

        self.SOH_MLP = MLP(in_channel=SOH_output_size, out_channel=configs.c_out, configs=configs)
        self.RUL_MLP = MLP(in_channel=RUL_output_size, out_channel=configs.c_out, configs=configs)

    def forward(self, x):

        if self.task == 'SOH':
            SOH_out = self.Backbone(x)
            SOH_out = self.xlstm(SOH_out)
            SOH_out = self.SOH_network(SOH_out)
            SOH_out = SOH_out.contiguous().view(SOH_out.size(0), -1)
            SOH_out = self.SOH_MLP(SOH_out)
            output = SOH_out.view(SOH_out.shape[0], 1, -1)

        elif self.task == 'RUL':
            RUL_out = self.Backbone(x)
            RUL_out = self.xlstm(RUL_out)
            RUL_out = self.RUL_network(RUL_out)
            RUL_out = RUL_out.contiguous().view(RUL_out.size(0), -1)
            RUL_out = self.RUL_MLP(RUL_out)
            output = RUL_out.view(RUL_out.shape[0], 1, -1)

        return output

class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(configs.d_model, configs.d_model, num_layers=1 , batch_first=True)
    def forward(self, x):
        out, _ = self.gru(x)
        return out

class SOHBlock(nn.Module):
    def __init__(self, configs):
        super(SOHBlock, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda:{}'.format(self.configs.gpu))
        self.Attention = self._build_model().to(self.device)
        self.feedforward = nn.Sequential(nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                                         nn.ReLU(),
                                         nn.Dropout(configs.dropout),
                                        nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),)
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
    def _build_model(self):
        model_dict = {
            'ACmix': ACmix1D,
            'CBAM': CBAM,
            'EMSA': EMSA,
            'PSA': SequentialPolarizedSelfAttention,
            'S2A': S2Attention,
            'SGE': SpatialGroupEnhance,
            'SKA': SKAttention,
            'TA': TripletAttention,
            'SSA': SparseSelfAttention,
            'Prob':Prob_AttentionLayer
        }
        model = model_dict[self.configs.soh_att](self.configs).float()

        if self.configs.use_multi_gpu and self.configs.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def forward(self, x):
        out = self.Attention(x)
        out = self.norm(x + out)
        out = self.norm(out + self.feedforward(out))
        out = self.dropout(out)
        return out

class RULBlock(nn.Module):
    def __init__(self, configs):
        super(RULBlock, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda:{}'.format(self.configs.gpu))
        self.Attention = self._build_model().to(self.device)
        self.feedforward = nn.Sequential(nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                                         nn.ReLU(),
                                         nn.Dropout(configs.dropout),
                                         nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),)
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.conv1 = nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, dilation=1,padding='same')
    def _build_model(self):
        model_dict = {
            'ACmix': ACmix1D,
            'CBAM': CBAM,
            'EMSA': EMSA,
            'PSA': SequentialPolarizedSelfAttention,
            'S2A': S2Attention,
            'SGE': SpatialGroupEnhance,
            'SKA': SKAttention,
            'TA': TripletAttention,
            'SSA': SparseSelfAttention,
            'Prob':Prob_AttentionLayer
        }
        model = model_dict[self.configs.rul_att](self.configs).float()

        if self.configs.use_multi_gpu and self.configs.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def forward(self, x):
        out = self.Attention(x)
        out = self.norm(x + out)
        out = self.norm(out + self.feedforward(out))
        out = self.dropout(out)
        return out

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, configs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, configs.d_mlp)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(configs.dropout)
        self.fc2 = nn.Linear(configs.d_mlp, out_channel)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, configs):
        super(InceptionModule, self).__init__()
        in_channels = configs.c_out
        out_channels = configs.d_model
        out1 = out_channels//4
        mid2 = out_channels//8
        out2 = out_channels//4
        mid3 = out_channels//8
        out3 = out_channels//4
        out4 = out_channels//4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out1, kernel_size=1),
            nn.BatchNorm1d(out1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, mid2, kernel_size=1),
            nn.BatchNorm1d(mid2),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid2, out2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out2),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, mid3, kernel_size=1),
            nn.BatchNorm1d(mid3),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid3, out3, kernel_size=5, padding=2),
            nn.BatchNorm1d(out3),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out4, kernel_size=1),
            nn.BatchNorm1d(out4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        output = torch.cat(outputs, 1)
        output = output.permute(0, 2, 1)
        return output
