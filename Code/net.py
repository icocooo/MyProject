import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from Code import gt_net_compound, gt_net_protein

# 设备配置：优先使用GPU加速计算
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class SelfAttention(nn.Module):
    """
    自注意力机制模块
    实现标准的Transformer多头自注意力机制，用于特征间的交互学习
    """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim  # 隐藏层维度
        self.n_heads = n_heads  # 注意力头数量
        assert hid_dim % n_heads == 0  # 确保维度能被头数整除

        # 线性变换层：Q、K、V矩阵投影
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)  # 输出投影层
        self.do = nn.Dropout(dropout)  # 注意力dropout
        # 缩放因子，用于稳定softmax计算
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        """
        前向传播过程
        参数:
            query: 查询张量 [batch_size, seq_len, hid_dim]
            key: 键张量 [batch_size, seq_len, hid_dim]
            value: 值张量 [batch_size, seq_len, hid_dim]
            mask: 可选掩码，用于屏蔽无效位置
        返回:
            注意力加权后的特征表示
        """
        bsz = query.shape[0]  # 批次大小

        # 线性投影得到Q、K、V
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 多头注意力机制：重塑张量维度
        # 原始: [batch_size, seq_len, hid_dim]
        # 变换: [batch_size, n_heads, seq_len, hid_dim//n_heads]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # 计算注意力能量分数: Q * K^T / sqrt(d_k)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 应用掩码（如需要）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 计算注意力权重: softmax归一化
        attention = self.do(F.softmax(energy, dim=-1))

        # 应用注意力权重到V值
        x = torch.matmul(attention, V)

        # 重塑回原始维度: [batch_size, seq_len, hid_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # 最终线性投影
        x = self.fc(x)
        return x


class MDGTDTInet(nn.Module):
    """
    多模态图Transformer药物-靶点相互作用预测网络
    主要功能: 预测化合物与蛋白质之间的结合亲和力
    """

    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1):
        super(MDGTDTInet, self).__init__()

        # 维度配置参数
        self.compound_dim = compound_dim  # 化合物特征维度
        self.protein_dim = protein_dim  # 蛋白质特征维度
        self.n_layers = gt_layers  # GraphTransformer层数
        self.n_heads = gt_heads  # 注意力头数

        # 交叉注意力模块: 用于蛋白质多模态特征融合
        self.crossAttention = SelfAttention(hid_dim=self.compound_dim, n_heads=1, dropout=0.2)

        # 双流图Transformer编码器
        # 化合物图编码器: 处理分子结构图
        self.compound_gt = gt_net_compound.GraphTransformer(
            device, n_layers=gt_layers, node_dim=44, edge_dim=10,
            hidden_dim=compound_dim, out_dim=compound_dim, n_heads=gt_heads,
            in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8
        )

        # 蛋白质图编码器: 处理蛋白质结构图
        self.protein_gt = gt_net_protein.GraphTransformer(
            device, n_layers=gt_layers, node_dim=41, edge_dim=5,
            hidden_dim=protein_dim, out_dim=protein_dim, n_heads=gt_heads,
            in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8
        )

        # 蛋白质特征处理层
        self.protein_embedding_fc = nn.Linear(320, self.protein_dim)  # ESM嵌入投影
        self.protein_fc = nn.Linear(self.protein_dim * 2, self.protein_dim)  # 特征融合

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # 联合注意力权重矩阵: 学习化合物-蛋白质相互作用
        self.joint_attn_prot = nn.Linear(compound_dim, compound_dim)  # 蛋白质侧
        self.joint_attn_comp = nn.Linear(compound_dim, compound_dim)  # 化合物侧

        self.modal_fc = nn.Linear(protein_dim * 2, protein_dim)  # 模态融合
        self.fc_out = nn.Linear(compound_dim, out_dim)  # 输出层

        # 分类器: 多层感知机进行最终预测
        self.classifier = nn.Sequential(
            nn.Linear(compound_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)  # 最终输出维度
        )

    def dgl_split(self, bg, feats):
        """
        DGL图批次数据分割函数
        将批处理的图数据重新组织为规整的张量格式，便于后续注意力计算

        参数:
            bg: 批处理的DGL图对象
            feats: 节点特征张量
        返回:
            重新组织后的特征张量 [batch_size, max_nodes, feat_dim]
        """
        # 计算批次中最大节点数（用于填充）
        max_num_nodes = int(bg.batch_num_nodes().max())

        # 创建批次索引: 为每个节点分配对应的图索引
        batch = torch.cat([
            torch.full((1, x.type(torch.int)), y) for x, y in
            zip(bg.batch_num_nodes(), range(bg.batch_size))
        ], dim=1).reshape(-1).type(torch.long).to(bg.device)

        # 计算累积节点数，用于索引计算
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])

        # 创建全局到局部索引的映射
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

        # 创建填充后的输出张量
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)  # 用0填充
        out[idx] = feats  # 将实际特征填入对应位置

        # 重塑为批次格式
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, compound_graph, protein_graph, protein_embedding):
        """
        前向传播过程
        参数:
            compound_graph: 化合物分子图数据
            protein_graph: 蛋白质结构图数据
            protein_embedding: 蛋白质序列嵌入特征
        返回:
            预测的结合亲和力分数
        """
        # 第一阶段: 化合物特征提取
        # 通过图Transformer编码器提取化合物图特征
        print()
        compound_feat = self.compound_gt(compound_graph)
        max_num_nodes = int(compound_graph.batch_num_nodes().max())
        print(compound_graph.batch_num_nodes())
        print(max_num_nodes)
        print(compound_feat.shape)
        # 重新组织特征张量为批次格式
        compound_feat_x = self.dgl_split(compound_graph, compound_feat)
        compound_feats = compound_feat_x
        print(compound_feats.shape)
        sys.exit(0)

        # 第二阶段: 蛋白质特征提取与融合
        # 提取蛋白质图结构特征
        protein_feat = self.protein_gt(protein_graph)
        protein_feat_x = self.dgl_split(protein_graph, protein_feat)

        # 处理蛋白质序列嵌入特征
        protein_embedding = self.protein_embedding_fc(protein_embedding)

        # 交叉注意力融合: 使用序列嵌入作为查询，关注图结构特征
        protein_feats = self.crossAttention(protein_embedding, protein_feat_x, protein_feat_x)

        # 第三阶段: 化合物-蛋白质相互作用建模
        # 计算原子-残基级别的相互作用权重矩阵
        # 使用爱因斯坦求和约定进行高效张量运算
        inter_comp_prot = self.sigmoid(
            torch.einsum('bij,bkj->bik',
                         self.joint_attn_prot(self.relu(protein_feats)),
                         self.joint_attn_comp(self.relu(compound_feats)))
        )

        # 归一化相互作用权重: 确保每对相互作用的权重和为1
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1 / inter_comp_prot_sum)

        # 第四阶段: 联合嵌入表示学习
        # 计算蛋白质和化合物特征的外积，捕获所有可能的相互作用对
        cp_embedding = self.tanh(
            torch.einsum('bij,bkj->bikj', protein_feats, compound_feats)
        )

        # 基于相互作用权重进行加权聚合，得到最终的联合嵌入
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # 第五阶段: 最终预测
        # 通过分类器网络预测结合亲和力
        x = self.classifier(cp_embedding)

        return x