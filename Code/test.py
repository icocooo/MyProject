import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import data_util1 as du
from Code import ProteinGNN
from Code.MolecularGNN import MolecularGNN
import csv
def compounds_feature_extract():
    """测试两个GNN模型"""

    # 初始化模型
    mol_gnn = MolecularGNN(node_dim=5, output_dim=512)

    # 模拟输入数据（基于预处理模块的输出格式）
    # 小分子数据
    path = '../Data/davis.csv'
    compounds = np.loadtxt(path, delimiter=',', skiprows=1,usecols=(1),max_rows=50,dtype='str')
    print(compounds)
    for i, compound_smiles in enumerate(compounds):
        print(f"\n--- 测试第 {i + 1} 个化合物 ---")
        print(f"SMILES: {compound_smiles}")

        try:
            # 1. 将SMILES转换为分子对象
            mol = Chem.MolFromSmiles(compound_smiles)
            if mol is None:
                print("无法解析SMILES，跳过")
                continue

            # 2. 添加氢原子并生成3D坐标
            mol = Chem.AddHs(mol)  # 添加氢原子
            AllChem.EmbedMolecule(mol)  # 生成3D坐标
            AllChem.MMFFOptimizeMolecule(mol)  # 能量最小化

            # 3. 转换为图结构
            graph = du.ligand_to_graph(mol)
            # 4. 转换为PyTorch张量
            node_features = torch.tensor(graph["node_feat"], dtype=torch.float32)
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            print(node_features.shape)
            print(f"节点数: {node_features.shape[0]}")
            print(f"边数: {edge_index.shape[1]}")

            # 5. 创建批处理索引（单图情况）
            batch = torch.zeros(node_features.shape[0], dtype=torch.long)

            # 6. 前向传播
            mol_features, mol_node_features = mol_gnn(node_features, edge_index, batch)

            print(f"分子特征形状: {mol_features.shape}")
            print(f"分子节点特征形状: {mol_node_features.shape}")

            # 7. 测试注意力权重
            attention_weights = mol_gnn.get_attention_weights(node_features, edge_index)
            print(f"注意力权重形状: {attention_weights.shape}")
            print(f"注意力权重范围: {attention_weights.min():.4f} - {attention_weights.max():.4f}")

        except Exception as e:
            print(f"处理化合物时出错: {e}")
            continue

        return mol_features
    # 前向传播
    # mol_features, mol_node_features = mol_gnn(mol_node_feat, mol_edge_index)

    # print(f"小分子特征形状: {mol_features.shape}")
    # print(f"小分子节点特征形状: {mol_node_features.shape}")
    #
    #
    # return mol_features, protein_features
def protein_feature_extract():
    path = '../Data/3HTB.pdb'
    mol = Chem.MolFromPDBFile(path)
    graph = du.protein_to_graph(mol)
    x = graph["node_feat"]

    model = ProteinGNN.ProteinFeatureExtractor(
        node_dim=x.shape[1],
        hidden_dim=128,           # 可以使用较小的维度
        output_dim=512,
        use_3d=False,             # 关键：禁用3D处理
        pooling_method='mean'     # 使用简单的池化方法
    )
    x= torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    result = model(x,edge_index)
    print(len(result[0]))

# 运行测试
if __name__ == "__main__":
    x=os.listdir('./')
    print(x)