import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Graph Transformer 层 ====================
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, g):
        # 计算注意力分数
        g.apply_edges(self.src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(self.scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(self.imp_exp_attn('score', 'proj_e'))
        g.apply_edges(self.out_edge_features('score'))
        g.apply_edges(self.exp('score'))

        # 消息传递
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def src_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

        return func

    def scaling(self, field, scale_constant):
        def func(edges):
            return {field: ((edges.data[field]) / scale_constant)}

        return func

    def imp_exp_attn(self, implicit_attn, explicit_edge):
        def func(edges):
            return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

        return func

    def out_edge_features(self, edge_feat):
        def func(edges):
            return {'e_out': edges.data[edge_feat]}

        return func

    def exp(self, field):
        def func(edges):
            return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

        return func

    def forward(self, g, h, e):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + 1e-6)
        e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=True, batch_norm=False, residual=True,
                 use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        # FFN
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

    def forward(self, g, h, e):
        h_in1, e_in1 = h, e

        # 多头注意力
        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h, e = h_in1 + h, e_in1 + e

        if hasattr(self, 'layer_norm1_h'):
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        h_in2, e_in2 = h, e

        # FFN
        h = F.relu(self.FFN_h_layer1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        e = F.relu(self.FFN_e_layer1(e))
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h, e = h_in2 + h, e_in2 + e

        if hasattr(self, 'layer_norm2_h'):
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        return h, e


# ==================== 简化的图编码器 ====================
class SimpleGraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads, dropout=0.1):
        super().__init__()
        self.device = device

        # 输入投影
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, True, False, True)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        g = g.to(self.device)

        # 节点和边特征
        h = g.ndata['feats'].float().to(self.device)
        e = g.edata['feats'].float().to(self.device)

        # 特征投影
        h = self.node_embedding(h)
        e = self.edge_embedding(e)
        h = self.dropout_layer(h)

        # Transformer层
        for layer in self.layers:
            h, e = layer(g, h, e)

        # 图级读出：平均池化
        h_graph = dgl.mean_nodes(g, 'h') if 'h' in g.ndata else torch.mean(h, dim=0, keepdim=True)

        # 如果h_graph是标量，添加批次维度
        if h_graph.dim() == 1:
            h_graph = h_graph.unsqueeze(0)

        return self.output_proj(h_graph)


# ==================== 完整的亲和力预测模型 ====================
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q, K, V = self.w_q(query), self.w_k(key), self.w_v(value)

        # 重塑为多头注意力
        head_dim = self.hid_dim // self.n_heads
        Q = Q.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        K = K.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)

        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)

        # 重塑回原始形状
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.hid_dim)
        x = self.fc(x)

        return x


class AffinityPredictionModel(nn.Module):
    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=8, out_dim=1):
        super().__init__()

        # 简化配置
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim

        # 化合物编码器
        self.compound_encoder = SimpleGraphTransformer(
            device, n_layers=gt_layers, node_dim=44, edge_dim=10,
            hidden_dim=compound_dim, out_dim=compound_dim, n_heads=gt_heads, dropout=0.2
        )

        # 蛋白质编码器
        self.protein_encoder = SimpleGraphTransformer(
            device, n_layers=gt_layers, node_dim=41, edge_dim=5,
            hidden_dim=protein_dim, out_dim=protein_dim, n_heads=gt_heads, dropout=0.2
        )

        # 蛋白质序列嵌入处理
        self.protein_embedding_fc = nn.Linear(320, protein_dim)

        # 交叉注意力
        self.cross_attention = SelfAttention(hid_dim=protein_dim, n_heads=4, dropout=0.1)

        # 相互作用建模
        self.interaction_fc = nn.Linear(compound_dim + protein_dim, 512)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, compound_graph, protein_graph, protein_embedding):
        # 化合物特征提取
        compound_feat = self.compound_encoder(compound_graph)  # [batch_size, compound_dim]

        # 蛋白质特征提取
        protein_feat = self.protein_encoder(protein_graph)  # [batch_size, protein_dim]

        # 蛋白质序列嵌入处理
        protein_embedding = self.protein_embedding_fc(protein_embedding)  # [batch_size, seq_len, protein_dim]

        # 对序列嵌入进行平均池化
        protein_seq_feat = torch.mean(protein_embedding, dim=1)  # [batch_size, protein_dim]

        # 融合蛋白质的图特征和序列特征
        protein_feat = protein_feat + protein_seq_feat  # 简单相加融合

        # 确保批次维度一致
        batch_size = compound_feat.shape[0]
        if protein_feat.shape[0] != batch_size:
            # 如果维度不匹配，使用广播
            if protein_feat.shape[0] == 1:
                protein_feat = protein_feat.expand(batch_size, -1)
            elif compound_feat.shape[0] == 1:
                compound_feat = compound_feat.expand(protein_feat.shape[0], -1)

        # 拼接特征
        combined_feat = torch.cat([compound_feat, protein_feat], dim=1)  # [batch_size, compound_dim + protein_dim]

        # 相互作用建模
        interaction_feat = self.relu(self.interaction_fc(combined_feat))
        interaction_feat = self.dropout(interaction_feat)

        # 最终预测
        affinity = self.classifier(interaction_feat)

        return affinity


# ==================== 使用示例 ====================
def create_single_graph(num_nodes, node_dim, edge_dim):
    """创建单个DGL图"""
    # 创建随机边
    num_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1))
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['feats'] = torch.randn(num_nodes, node_dim)
    g.edata['feats'] = torch.randn(num_edges, edge_dim)

    return g


def create_batch_graphs(batch_size, min_nodes=10, max_nodes=100, node_dim=44, edge_dim=10):
    """创建批次DGL图"""
    graphs = []
    for i in range(batch_size):
        num_nodes = torch.randint(min_nodes, max_nodes + 1, (1,)).item()
        g = create_single_graph(num_nodes, node_dim, edge_dim)
        graphs.append(g)

    return dgl.batch(graphs)


# 测试模型
if __name__ == "__main__":
    # 创建模型
    model = AffinityPredictionModel(
        compound_dim=128,
        protein_dim=128,
        gt_layers=3,
        gt_heads=8,
        out_dim=1
    ).to(device)

    print("模型创建成功!")

    # 创建示例数据
    batch_size = 2

    # 创建化合物图批次
    compound_graph = create_batch_graphs(
        batch_size=batch_size,
        min_nodes=20,
        max_nodes=50,
        node_dim=44,
        edge_dim=10
    )

    # 创建蛋白质图批次
    protein_graph = create_batch_graphs(
        batch_size=batch_size,
        min_nodes=50,
        max_nodes=100,
        node_dim=41,
        edge_dim=5
    )

    # 创建蛋白质序列嵌入
    protein_embedding = torch.randn(batch_size, 100, 320).to(device)  # [batch_size, seq_len, 320]

    print(f"化合物图批次: {compound_graph.batch_size}个图, 总节点数: {compound_graph.num_nodes()}")
    print(f"蛋白质图批次: {protein_graph.batch_size}个图, 总节点数: {protein_graph.num_nodes()}")
    print(f"蛋白质嵌入形状: {protein_embedding.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        try:
            affinity_score = model(compound_graph, protein_graph, protein_embedding)
            print(f"✅ 预测成功!")
            print(f"亲和力分数形状: {affinity_score.shape}")
            print(f"亲和力分数: {affinity_score}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback

            traceback.print_exc()