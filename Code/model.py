import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.MolecularGNN import MolecularGNN
from Code.ProteinGNN import ProteinFeatureExtractor


class AffinityPredictor(nn.Module):
    """
    亲和力预测模型
    接收小分子特征和蛋白特征，预测两者的结合亲和力
    """

    def __init__(self,
                 protein_feature_dim=512,
                 molecular_feature_dim=512,
                 hidden_dim=256,
                 num_interaction_layers=3,
                 dropout=0.2):
        super(AffinityPredictor, self).__init__()

        self.protein_feature_dim = protein_feature_dim
        self.molecular_feature_dim = molecular_feature_dim
        self.hidden_dim = hidden_dim

        # 特征投影层 - 将特征映射到统一维度
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.molecular_proj = nn.Sequential(
            nn.Linear(molecular_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 特征交互模块
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(hidden_dim * 2, hidden_dim, dropout)
            for _ in range(num_interaction_layers)
        ])

        # 注意力融合机制
        self.attention_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 亲和力预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 输出单个亲和力值
        )

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, protein_features, molecular_features):
        """
        前向传播
        Args:
            protein_features: 蛋白特征 [batch_size, protein_feature_dim]
            molecular_features: 小分子特征 [batch_size, molecular_feature_dim]
        Returns:
            affinity: 预测的亲和力分数 [batch_size, 1]
            attention_weights: 注意力权重（用于可解释性）
        """
        # 特征投影
        p_feat = self.protein_proj(protein_features)  # [batch_size, hidden_dim]
        m_feat = self.molecular_proj(molecular_features)  # [batch_size, hidden_dim]

        # 初始特征拼接
        combined = torch.cat([p_feat, m_feat], dim=1)  # [batch_size, hidden_dim * 2]

        # 特征交互
        interaction_features = combined
        for interaction_block in self.interaction_blocks:
            interaction_features = interaction_block(interaction_features)

        # 注意力融合
        attention_weights = self.attention_fusion(interaction_features)  # [batch_size, 1]
        weighted_features = interaction_features * attention_weights

        # 残差连接
        fused_features = combined + weighted_features

        # 亲和力预测
        affinity = self.prediction_head(fused_features)

        return affinity, attention_weights


class InteractionBlock(nn.Module):
    """特征交互块"""

    def __init__(self, input_dim, hidden_dim, dropout):
        super(InteractionBlock, self).__init__()

        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)  # 输出维度与输入相同
        )

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接
        residual = x
        # 交互变换
        interaction_out = self.interaction_net(x)
        # 添加残差和层归一化
        output = self.layer_norm(residual + self.dropout(interaction_out))
        return output


class CrossAttentionInteraction(nn.Module):
    """交叉注意力交互模块（可选）"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionInteraction, self).__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, protein_feat, molecular_feat):
        # 添加序列维度
        p_seq = protein_feat.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        m_seq = molecular_feat.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 蛋白到小分子的交叉注意力
        m_attended, m_weights = self.cross_attention(
            query=m_seq,
            key=p_seq,
            value=p_seq
        )

        # 残差连接和层归一化
        m_enhanced = self.norm1(m_seq + m_attended)

        # 前馈网络
        m_output = self.norm2(m_enhanced + self.ffn(m_enhanced))

        return m_output.squeeze(1), m_weights


# 完整的端到端模型封装
class DrugTargetAffinityModel(nn.Module):
    """完整的药物-靶标亲和力预测模型"""

    def __init__(self,
                 protein_model_config=None,
                 molecular_model_config=None,
                 affinity_model_config=None):
        super(DrugTargetAffinityModel, self).__init__()

        # 默认配置
        protein_model_config = protein_model_config or {}
        molecular_model_config = molecular_model_config or {}
        affinity_model_config = affinity_model_config or {}

        # 初始化特征提取器
        self.protein_extractor = ProteinFeatureExtractor(**protein_model_config)
        self.molecular_extractor = MolecularGNN(**molecular_model_config)

        # 初始化亲和力预测器
        self.affinity_predictor = AffinityPredictor(**affinity_model_config)

        # 训练模式设置
        self.freeze_protein_extractor = False
        self.freeze_molecular_extractor = False

    def forward(self, protein_data, molecular_data):
        """
        完整的前向传播
        Args:
            protein_data: 蛋白图数据
            molecular_data: 小分子图数据
        Returns:
            affinity: 亲和力预测值
            protein_features: 蛋白特征（可选）
            molecular_features: 小分子特征（可选）
        """
        # 提取特征
        with torch.set_grad_enabled(not self.freeze_protein_extractor):
            protein_features = self.protein_extractor(
                x=protein_data.x,
                edge_index=protein_data.edge_index,
                batch=protein_data.batch
            )

        with torch.set_grad_enabled(not self.freeze_molecular_extractor):
            molecular_features, node_features = self.molecular_extractor(
                x=molecular_data.x,
                edge_index=molecular_data.edge_index,
                batch=molecular_data.batch
            )

        # 预测亲和力
        affinity, attention_weights = self.affinity_predictor(
            protein_features, molecular_features
        )

        return {
            'affinity': affinity,
            'protein_features': protein_features,
            'molecular_features': molecular_features,
            'attention_weights': attention_weights,
            'molecular_node_features': node_features
        }

    def set_training_mode(self, freeze_protein=False, freeze_molecular=False):
        """设置训练模式"""
        self.freeze_protein_extractor = freeze_protein
        self.freeze_molecular_extractor = freeze_molecular

        # 设置参数requires_grad
        for param in self.protein_extractor.parameters():
            param.requires_grad = not freeze_protein

        for param in self.molecular_extractor.parameters():
            param.requires_grad = not freeze_molecular


# 训练工具类
class AffinityTrainer:
    """亲和力预测训练器"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()  # 回归任务
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # 移动到设备
            protein_data = batch['protein'].to(self.device)
            molecular_data = batch['molecular'].to(self.device)
            affinity_labels = batch['affinity'].to(self.device).float()

            # 前向传播
            outputs = self.model(protein_data, molecular_data)
            predicted_affinity = outputs['affinity'].squeeze()

            # 计算损失
            loss = self.criterion(predicted_affinity, affinity_labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, protein_data, molecular_data):
        """预测亲和力"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(protein_data, molecular_data)
            return outputs['affinity'].squeeze().cpu().numpy()


# 使用示例
if __name__ == "__main__":
    # 配置模型
    model = DrugTargetAffinityModel(
        protein_model_config={'output_dim': 512},
        molecular_model_config={'output_dim': 512},
        affinity_model_config={'hidden_dim': 256}
    )

    # 创建训练器
    trainer = AffinityTrainer(model)

    # 训练模式设置示例
    model.set_training_mode(freeze_protein=True, freeze_molecular=False)  # 只训练小分子部分

    print("亲和力预测模型创建完成！")