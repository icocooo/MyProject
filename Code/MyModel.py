import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from typing import Optional, Dict
import math

from torch.nn.utils.rnn import pad_sequence

from Code.compound_net import compound_net
from Code.protein_net import protein_net


class Model_Demo(torch.nn.Module):
    """DeepDTAGen主模型 - 多任务药物靶点亲和力预测和分子生成"""

    def __init__(self):
        super(Model_Demo, self).__init__()
        self.compound_net = compound_net()
        self.protein_net = protein_net()
        self.fc = FC(output_dim=128,n_output=1,dropout=0.3)
    def forward(self, compound_graph,protein_graph):

        compound_feat = self.compound_net(compound_graph)
        protein_feat = self.protein_net(protein_graph)
        # 亲和力预测
        Pridection = self.fc(compound_feat, protein_feat)
        return Pridection
class FC(torch.nn.Module):
    """全连接预测头，用于亲和力预测"""

    def __init__(self, output_dim, n_output, dropout):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(output_dim * 2, 1024),  # 拼接药物和蛋白质特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output)  # 输出亲和力预测
        )

    def forward(self, Drug_Features, Protein_Features):
        """亲和力预测前向传播"""
        Combined = torch.cat((Drug_Features, Protein_Features), 1)  # 特征拼接
        Pridection = self.FC_layers(Combined)
        return Pridection
