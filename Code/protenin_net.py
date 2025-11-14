import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from torch.cuda.amp import custom_fwd

from Code.DTIDataset import DTIDataset


class protein_net(nn.Module):
    def __init__(self, residue_feat_dim=41, lap_pe_dim=8, contact_feat_dim=5,
                 hidden_size=64, output_size=128):
        """
        蛋白图特征提取模块

        Args:
            residue_feat_dim: 残基特征维度 (41维，来自ndata['feats'])
            lap_pe_dim: 拉普拉斯位置编码维度 (8维，来自ndata['lap_pos_enc'])
            contact_feat_dim: 残基接触特征维度 (5维，来自edata['feats'])
            hidden_size: 隐藏层维度
            output_size: 输出特征维度 (默认128)
        """
        super(protein_net, self).__init__()

        # 计算融合后的节点特征维度：残基特征(41) + 拉普拉斯编码(8) = 49维
        fused_node_feats = residue_feat_dim + lap_pe_dim

        # 节点特征编码层（融合残基特征和拉普拉斯编码）
        self.node_encoder = nn.Linear(fused_node_feats, hidden_size)

        # 边特征编码层（处理残基接触特征）
        self.edge_encoder = nn.Linear(contact_feat_dim, hidden_size)

        # 图卷积层
        self.conv1 = dglnn.GraphConv(hidden_size, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, hidden_size)
        self.conv3 = dglnn.GraphConv(hidden_size, hidden_size)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # 注意力池化层
        self.gate_nn = nn.Linear(hidden_size, 1)
        self.pool = dglnn.GlobalAttentionPooling(self.gate_nn)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, output_size)
        )

        # 激活函数
        self.activation = nn.ReLU()

    def fuse_node_features(self, g):
        """
        融合残基特征和拉普拉斯位置编码
        根据你的数据格式：ndata['feats'] + ndata['lap_pos_enc']
        """
        residue_features = g.ndata['feats'].float()  # 转换为float32
        laplacian_pe = g.ndata['lap_pos_enc']

        # 拼接残基特征和拉普拉斯编码
        fused_features = torch.cat([residue_features, laplacian_pe], dim=1)
        return fused_features

    def process_edge_features(self, g):
        """
        处理边特征（残基接触特征）
        """
        contact_features = g.edata['feats'].float()  # 转换为float32
        return contact_features

    @custom_fwd
    def forward(self, g):
        """
        Args:
            g: 蛋白DGL图对象，包含：
                - ndata['feats']: 41维残基特征 (float64)
                - ndata['lap_pos_enc']: 8维拉普拉斯位置编码 (float32)
                - edata['feats']: 5维残基接触特征 (float64)

        Returns:
            torch.Tensor: 128维蛋白特征向量
        """
        # 融合节点特征
        fused_node_feats = self.fuse_node_features(g)
        edge_feats = self.process_edge_features(g)

        # 编码节点和边特征
        h = self.activation(self.node_encoder(fused_node_feats))
        e = self.activation(self.edge_encoder(edge_feats))

        # 图卷积操作
        h = self.activation(self.bn1(self.conv1(g, h, edge_weight=e.mean(dim=1))))
        h = self.activation(self.bn2(self.conv2(g, h, edge_weight=e.mean(dim=1))))
        h = self.activation(self.bn3(self.conv3(g, h, edge_weight=e.mean(dim=1))))

        # 全局池化得到图级表示
        graph_feat = self.pool(g, h)

        # 输出128维特征
        output = self.output_layer(graph_feat)

        return output


# 测试代码
if __name__ == "__main__":
    # 创建测试用的蛋白图数据（模拟你的数据格式）
    dataset = DTIDataset()
    g = dataset[0][1]

    print("蛋白图结构信息:")
    print(f"节点数: {g.num_nodes()}")
    print(f"边数: {g.num_edges()}")
    print(f"残基特征形状: {g.ndata['feats'].shape}")
    print(f"拉普拉斯编码形状: {g.ndata['lap_pos_enc'].shape}")
    print(f"接触特征形状: {g.edata['feats'].shape}")

    # 测试模型
    model = protein_net(
        residue_feat_dim=41,
        lap_pe_dim=8,
        contact_feat_dim=5,
        output_size=128
    )

    # 提取特征
    with torch.no_grad():
        features = model(g)
        print(f"\n输出特征形状: {features.shape}")
        print(f"特征维度: {features.shape[-1]}")