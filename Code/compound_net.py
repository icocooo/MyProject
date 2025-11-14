import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from torch.cuda.amp import custom_fwd

from Code.DTIDataset import DTIDataset


class compound_net(nn.Module):
    def __init__(self, atom_feat_dim=44, lap_pe_dim=8, bond_feat_dim=10, hidden_size=64, output_size=128):
        """
        针对你的数据定制的小分子图特征提取模块

        Args:
            atom_feat_dim: 原子特征维度 (44维，来自你的ndata['atom'])
            lap_pe_dim: 拉普拉斯位置编码维度 (8维，来自你的ndata['lap_pos_enc'])
            bond_feat_dim: 键特征维度 (10维，来自你的edata['bond'])
            hidden_size: 隐藏层维度
            output_size: 输出特征维度 (默认128)
        """
        super(compound_net, self).__init__()

        # 计算融合后的节点特征维度：atom(44) + lap_pos_enc(8) = 52维
        fused_node_feats = atom_feat_dim + lap_pe_dim

        # 节点特征编码层（融合原子特征和拉普拉斯编码）
        self.node_encoder = nn.Linear(fused_node_feats, hidden_size)

        # 边特征编码层（处理bond特征）
        self.edge_encoder = nn.Linear(bond_feat_dim, hidden_size)

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
        融合原子特征和已有的拉普拉斯位置编码
        根据你的数据格式：ndata['atom'] + ndata['lap_pos_enc']
        """
        atom_features = g.ndata['atom'].float()  # 转换为float32，与lap_pos_enc一致
        laplacian_pe = g.ndata['lap_pos_enc']

        # 拼接原子特征和拉普拉斯编码
        fused_features = torch.cat([atom_features, laplacian_pe], dim=1)
        return fused_features

    def process_edge_features(self, g):
        """
        处理边特征（bond特征）
        """
        bond_features = g.edata['bond'].float()  # 转换为float32
        return bond_features

    @custom_fwd
    def forward(self, g):
        """
        Args:
            g: 你的DGL图对象，包含：
                - ndata['atom']: 44维原子特征 (float64)
                - ndata['lap_pos_enc']: 8维拉普拉斯位置编码 (float32)
                - edata['bond']: 10维键特征 (int64)

        Returns:
            torch.Tensor: 128维分子特征向量
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

# 使用示例和测试
if __name__ == "__main__":

    # 测试模型
    model = compound_net(
        atom_feat_dim=44,
        lap_pe_dim=8,
        bond_feat_dim=10,
        output_size=128
    )
    dataset = DTIDataset()
    test_graph = dataset[0][1]
    print(test_graph)
    print("图结构信息:")
    print(f"节点数: {test_graph.num_nodes()}")
    print(f"边数: {test_graph.num_edges()}")
    print(f"原子特征形状: {test_graph.ndata['atom'].shape}")
    print(f"拉普拉斯编码形状: {test_graph.ndata['lap_pos_enc'].shape}")
    print(f"键特征形状: {test_graph.edata['bond'].shape}")

    # 提取特征
    with torch.no_grad():
        features = model(test_graph)
        print(f"\n输出特征形状: {features.shape}")
        print(f"特征维度: {features.shape[-1]}")