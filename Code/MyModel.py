import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GlobalAttentionPooling
import matplotlib
class DualStreamDTIModel(nn.Module):
    def __init__(self,
                 compound_node_dim=44,
                 compound_edge_dim=10,
                 protein_node_dim=41,
                 protein_edge_dim=5,
                 hidden_dim=128,
                 n_heads=8,
                 n_layers=3,
                 dropout=0.2):
        super(DualStreamDTIModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # 化合物图编码器
        self.compound_encoders = nn.ModuleList()
        self.compound_encoders.append(
            GraphConv(compound_node_dim, hidden_dim)
        )
        for _ in range(n_layers - 1):
            self.compound_encoders.append(
                GraphConv(hidden_dim, hidden_dim)
            )

        # 蛋白质图编码器
        self.protein_encoders = nn.ModuleList()
        self.protein_encoders.append(
            GraphConv(protein_node_dim, hidden_dim)
        )
        for _ in range(n_layers - 1):
            self.protein_encoders.append(
                GraphConv(hidden_dim, hidden_dim)
            )

        # 注意力池化层
        self.compound_pool = GlobalAttentionPooling(
            nn.Linear(hidden_dim, 1)
        )
        self.protein_pool = GlobalAttentionPooling(
            nn.Linear(hidden_dim, 1)
        )

        # 交叉注意力机制
        self.cross_attention = MultiHeadCrossAttention(
            hidden_dim, n_heads, dropout
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, compound_graph, protein_graph):
        # 提取化合物特征
        compound_feat = compound_graph.ndata['atom'].float()
        for i, encoder in enumerate(self.compound_encoders):
            compound_feat = encoder(compound_graph, compound_feat)
            if i < len(self.compound_encoders) - 1:
                compound_feat = F.relu(compound_feat)
                compound_feat = self.dropout(compound_feat)

        # 提取蛋白质特征
        protein_feat = protein_graph.ndata['feats'].float()
        for i, encoder in enumerate(self.protein_encoders):
            protein_feat = encoder(protein_graph, protein_feat)
            if i < len(self.protein_encoders) - 1:
                protein_feat = F.relu(protein_feat)
                protein_feat = self.dropout(protein_feat)

        # 全局池化得到图级表示
        compound_global = self.compound_pool(compound_graph, compound_feat)
        protein_global = self.protein_pool(protein_graph, protein_feat)

        # 交叉注意力交互
        compound_attended, protein_attended = self.cross_attention(
            compound_feat, protein_feat, compound_graph, protein_graph
        )

        # 多层级特征融合
        compound_final = torch.cat([
            compound_global,
            torch.mean(compound_attended, dim=1),
            torch.max(compound_attended, dim=1)[0]
        ], dim=1)

        protein_final = torch.cat([
            protein_global,
            torch.mean(protein_attended, dim=1),
            torch.max(protein_attended, dim=1)[0]
        ], dim=1)

        # 最终预测
        combined = torch.cat([compound_final, protein_final], dim=1)
        affinity = self.predictor(combined)

        return affinity.squeeze(-1)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(MultiHeadCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.compound_query = nn.Linear(hidden_dim, hidden_dim)
        self.compound_key = nn.Linear(hidden_dim, hidden_dim)
        self.compound_value = nn.Linear(hidden_dim, hidden_dim)

        self.protein_query = nn.Linear(hidden_dim, hidden_dim)
        self.protein_key = nn.Linear(hidden_dim, hidden_dim)
        self.protein_value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_compound = nn.Linear(hidden_dim, hidden_dim)
        self.fc_protein = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, compound_feat, protein_feat, compound_graph, protein_graph):
        batch_size = compound_feat.size(0)

        # 线性投影
        Q_compound = self.compound_query(compound_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        K_protein = self.protein_key(protein_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        V_protein = self.protein_value(protein_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # 计算注意力分数
        energy_compound = torch.matmul(Q_compound, K_protein.transpose(-2, -1)) / self.scale
        attention_compound = F.softmax(energy_compound, dim=-1)
        attention_compound = self.dropout(attention_compound)

        # 应用注意力到蛋白质特征
        compound_attended = torch.matmul(attention_compound, V_protein)
        compound_attended = compound_attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        compound_attended = self.fc_compound(compound_attended)

        # 蛋白质到化合物的对称注意力
        Q_protein = self.protein_query(protein_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        K_compound = self.compound_key(compound_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        V_compound = self.compound_value(compound_feat).view(
            batch_size, -1, self.n_heads, self.head_dim
        ).transpose(1, 2)

        energy_protein = torch.matmul(Q_protein, K_compound.transpose(-2, -1)) / self.scale
        attention_protein = F.softmax(energy_protein, dim=-1)
        attention_protein = self.dropout(attention_protein)

        protein_attended = torch.matmul(attention_protein, V_compound)
        protein_attended = protein_attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        protein_attended = self.fc_protein(protein_attended)

        return compound_attended, protein_attended