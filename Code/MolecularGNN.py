import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class MolecularGNN(nn.Module):
    """小分子图神经网络特征提取器"""

    def __init__(self, node_dim=6, hidden_dim=256, output_dim=512, num_layers=4, dropout=0.1):
        super(MolecularGNN, self).__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 初始节点特征投影
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # GCN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_channels, hidden_dim))

        # 批量归一化层
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 注意力机制用于图池化
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批索引 [num_nodes]
        Returns:
            molecular_features: 分子特征向量 [batch_size, output_dim]
            node_features: 节点级特征 [num_nodes, hidden_dim]
        """

        # 节点特征投影
        x = self.node_proj(x)

        # 多层GCN传播
        for i in range(self.num_layers):
            # GCN卷积
            x = self.convs[i](x, edge_index)
            # 批量归一化
            x = self.bns[i](x)
            # 激活函数
            x = F.relu(x)
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_features = x  # 保存节点级特征

        # 如果提供了batch信息，进行图池化
        if batch is not None:
            # 注意力池化
            attention_weights = torch.softmax(self.attention(x), dim=0)
            weighted_sum = torch.sum(x * attention_weights, dim=0)

            # 全局平均池化和最大池化
            global_mean = global_mean_pool(x, batch)
            global_max = global_max_pool(x, batch)

            # 拼接不同池化方式的结果
            pooled = torch.cat([global_mean, global_max], dim=1)

            # 最终投影
            molecular_features = self.output_proj(pooled)

            return molecular_features, node_features
        else:
            # 单图情况，直接返回节点特征
            return x, node_features

    def get_attention_weights(self, x, edge_index):
        """获取节点注意力权重，用于可视化"""
        x = self.node_proj(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        attention_weights = torch.softmax(self.attention(x), dim=0)
        return attention_weights.squeeze()