import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch_scatter import scatter_mean, scatter_max


class ProteinFeatureExtractor(nn.Module):
    """优化的蛋白质特征提取器（只输出全局特征）"""

    def __init__(self,
                 node_dim=6,
                 hidden_dim=512,
                 output_dim=1024,
                 num_layers=5,
                 dropout=0.2,
                 use_3d=True,
                 pooling_method='attention'):
        super(ProteinFeatureExtractor, self).__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_3d = use_3d
        self.pooling_method = pooling_method

        # 3D位置编码（如果使用3D信息）
        if self.use_3d:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, hidden_dim // 8),
                nn.ReLU()
            )
            node_input_dim = node_dim + hidden_dim // 8
        else:
            node_input_dim = node_dim

        # 初始节点特征投影
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GIN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
            self.convs.append(GINConv(gin_nn))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 多尺度特征融合（简化版）
        self.fusion_layers = min(3, num_layers)
        self.scale_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(self.fusion_layers)
        ])

        # 图池化机制（优化版）
        if pooling_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.fusion_layers, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Dropout(dropout)
            )
        elif pooling_method == 'multi_head_attention':
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * self.fusion_layers, hidden_dim // 4),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Dropout(dropout)
                ) for _ in range(4)
            ])

        # 输出投影层（优化）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.fusion_layers * 3, hidden_dim * 2),  # 3种池化组合
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout // 2)  # 输出层dropout减半
        )

        # 残差连接（简化）
        self.residual = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, positions=None, batch=None):
        """
        优化的前向传播，只返回蛋白质全局特征
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            positions: 3D坐标 [num_nodes, 3]
            batch: 批索引 [num_nodes]
        Returns:
            protein_features: 蛋白质特征向量 [batch_size, output_dim]
        """

        # 3D位置编码
        if self.use_3d and positions is not None:
            pos_encoding = self.pos_encoder(positions)
            x = torch.cat([x, pos_encoding], dim=1)

        # 初始节点特征投影
        x = self.node_proj(x)
        x_residual = self.residual(x)

        # 存储多尺度特征
        layer_features = []

        # GIN消息传递
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

            # 残差连接（只在第一层）
            if i == 0:
                x = x + x_residual

            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 保存最后几层特征
            if i >= self.num_layers - self.fusion_layers:
                layer_features.append(x)

        # 多尺度特征融合
        fused_features = []
        for i, feat in enumerate(layer_features):
            transformed = self.scale_fusions[i](feat)
            fused_features.append(transformed)

        # 拼接多尺度特征
        x_fused = torch.cat(fused_features, dim=1)

        # 图池化得到全局特征
        if batch is not None:
            if self.pooling_method == 'attention':
                pooled_features = self.attention_pooling(x_fused, batch)
            elif self.pooling_method == 'multi_head_attention':
                pooled_features = self.multi_head_attention_pooling(x_fused, batch)
            else:
                pooled_features = self.multi_pooling(x_fused, batch)

            # 最终投影
            protein_features = self.output_proj(pooled_features)
            return protein_features
        else:
            # 单图情况，手动池化
            global_mean = x_fused.mean(dim=0, keepdim=True)
            global_max, _ = x_fused.max(dim=0, keepdim=True)
            pooled = torch.cat([global_mean, global_max,
                                x_fused.std(dim=0, keepdim=True)], dim=1)
            return self.output_proj(pooled)

    def attention_pooling(self, x, batch):
        """优化的注意力池化"""
        attention_weights = F.softmax(self.attention(x), dim=0)
        weighted_sum = scatter_mean(x * attention_weights, batch, dim=0)

        global_mean = scatter_mean(x, batch, dim=0)
        global_max, _ = scatter_max(x, batch, dim=0)

        pooled = torch.cat([weighted_sum, global_mean, global_max], dim=1)
        return pooled

    def multi_head_attention_pooling(self, x, batch):
        """优化的多头注意力池化"""
        head_outputs = []
        for head in self.attention_heads:
            attention_weights = F.softmax(head(x), dim=0)
            weighted = scatter_mean(x * attention_weights, batch, dim=0)
            head_outputs.append(weighted)

        global_mean = scatter_mean(x, batch, dim=0)
        global_max, _ = scatter_max(x, batch, dim=0)

        head_outputs.extend([global_mean, global_max])
        pooled = torch.cat(head_outputs, dim=1)
        return pooled

    def multi_pooling(self, x, batch):
        """优化的多池化组合"""
        global_mean = scatter_mean(x, batch, dim=0)
        global_max, _ = scatter_max(x, batch, dim=0)
        global_std = torch.sqrt(scatter_mean((x - global_mean[batch]) ** 2, batch, dim=0) + 1e-8)

        pooled = torch.cat([global_mean, global_max, global_std], dim=1)
        return pooled