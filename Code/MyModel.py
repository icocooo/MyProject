import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from torch.utils.data import DataLoader

from Code.DTIDataset import DTIDataset


class DTI_Model(nn.Module):
    def __init__(self, compound_in_dim=74, protein_in_dim=128, hidden_dim=128):
        super(DTI_Model, self).__init__()

        # 化合物图编码器
        self.compound_conv1 = dglnn.GraphConv(compound_in_dim, hidden_dim)
        self.compound_conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)

        # 蛋白质图编码器
        self.protein_conv1 = dglnn.GraphConv(protein_in_dim, hidden_dim)
        self.protein_conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, compound_graph, protein_graph):
        # 化合物特征提取
        compound_feat = torch.randn(compound_graph.num_nodes(), 74)
        compound_feat = torch.relu(self.compound_conv1(compound_graph, compound_feat))
        compound_feat = torch.relu(self.compound_conv2(compound_graph, compound_feat))
        compound_global = dgl.mean_nodes(compound_graph, compound_feat)

        # 蛋白质特征提取
        protein_feat = torch.randn(protein_graph.num_nodes(), 128)
        protein_feat = torch.relu(self.protein_conv1(protein_graph, protein_feat))
        protein_feat = torch.relu(self.protein_conv2(protein_graph, protein_feat))
        protein_global = dgl.mean_nodes(protein_graph, protein_feat)

        # 拼接特征并预测
        combined = torch.cat([compound_global, protein_global], dim=1)
        output = self.predictor(combined)

        return output.squeeze()

def test_model():
    """测试模型是否能正常运行"""
    print("开始测试模型...")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    dataset = DTIDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)

    # 获取一个batch
    compound_graph, protein_graph, labels = next(iter(loader))
    print(f"数据加载成功: 化合物图{compound_graph.batch_size}, 蛋白质图{protein_graph.batch_size}")

    # 初始化模型
    model = DTI_Model().to(device)
    print("模型初始化成功")

    # 前向传播测试
    output = model(compound_graph, protein_graph)
    print(f"前向传播成功: 输出形状{output.shape}")

    print("测试完成！模型可以正常运行")


if __name__ == "__main__":
    test_model()