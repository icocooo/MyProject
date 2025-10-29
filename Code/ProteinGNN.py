import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch_scatter import scatter_mean, scatter_max


class ProteinFeatureExtractor(nn.Module):
    """简化版的蛋白质特征提取器"""

    def __init__(self,
                 node_dim=6,
                 hidden_dim=128,
                 output_dim=512,
                 num_layers=3,
                 dropout=0.2,
                 use_3d=False,
                 pooling_method='mean'):
        super(ProteinFeatureExtractor, self).__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_3d = use_3d
        self.pooling_method = pooling_method


        # 简化输入处理
        if self.use_3d:
            self.pos_encoder = nn.Linear(3, 8)  # 固定小维度
            node_input_dim = node_dim + 8
        else:
            node_input_dim = node_dim

        # 初始投影
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # 简化的GIN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(GINConv(gin_nn))

        # 简化池化机制
        if pooling_method == 'mean':
            self.pool_fn = global_mean_pool
            pool_output_dim = hidden_dim
        elif pooling_method == 'max':
            self.pool_fn = global_max_pool
            pool_output_dim = hidden_dim
        else:  # 简单组合
            self.pool_fn = None
            pool_output_dim = hidden_dim * 2

        # 简化输出层 - 固定维度避免计算错误
        self.output_proj = nn.Sequential(
            nn.Linear(pool_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, positions=None, batch=None):
        # 处理3D信息
        if self.use_3d and positions is not None:
            pos_encoding = self.pos_encoder(positions)
            x = torch.cat([x, pos_encoding], dim=1)

        # 初始投影
        x = F.relu(self.node_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GIN层
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 池化
        if batch is not None:
            if self.pool_fn is not None:
                x_pooled = self.pool_fn(x, batch)
            else:
                # 简单组合池化
                mean_pool = scatter_mean(x, batch, dim=0)
                max_pool, _ = scatter_max(x, batch, dim=0)
                x_pooled = torch.cat([mean_pool, max_pool], dim=1)
        else:
            # 单图情况
            if self.pool_fn is not None:
                x_pooled = x.mean(dim=0, keepdim=True)
            else:
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool, _ = x.max(dim=0, keepdim=True)
                x_pooled = torch.cat([mean_pool, max_pool], dim=1)

        # 输出投影
        return self.output_proj(x_pooled)