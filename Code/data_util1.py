"""
蛋白质和小分子预处理模块 - 原子级图结构版本
包含序列编码、图结构构建等功能
修改：将蛋白质图结构从残基级改为原子级
"""

import numpy as np
import rdkit.Chem as Chem
import networkx as nx
import os
import torch
from collections import OrderedDict
from rdkit.Chem import AllChem
import re

from build_vocab import WordVocab
def protein_atom_to_feature(atom):
    """将RDKit原子对象转换为特征向量 - 增强版"""
    features = [
        atom.GetAtomicNum(),              # 原子序数
        atom.GetDegree(),                 # 连接度
        atom.GetTotalNumHs(includeNeighbors=True),  # 氢原子数
        atom.GetImplicitValence(),        # 隐式价电子
        int(atom.GetIsAromatic()),        # 是否芳香族
        atom.GetFormalCharge(),           # 形式电荷
    ]
    return features
def protein_to_graph(protein):
    """将蛋白质分子转换为原子级图结构 - 使用你提供的代码"""
    # atom features
    atoms_feature_list = [protein_atom_to_feature(atom) for atom in protein.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype=np.float64)

    # 3D coordinates
    c = protein.GetConformer()
    coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)]
                   for atom_index in range(protein.GetNumAtoms())]
    node_positions = np.array(coordinates, dtype=np.float64)

    # bond edges
    if protein.GetNumBonds() > 0:
        edge_list = []
        for bond in protein.GetBonds():
            atom_u, atom_v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((atom_u, atom_v))
            edge_list.append((atom_v, atom_u))  # 无向图，双向连接
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = {
        "node_feat": node_features,
        "num_nodes": node_features.shape[0],
        "edge_index": edge_index,
        "node_positions": node_positions
    }

    return graph
# 原子类型字典（用于识别SMILES中的原子）

# 独立函数版本（不依赖类）
def smiles_to_label(smiles, max_length=100, charset_type='canonical'):
    """独立函数：SMILES转标签编码"""
    charset = CHARCANSMISET if charset_type == 'canonical' else CHARISOSMISET

    X = np.zeros(max_length)

    for i, ch in enumerate(smiles[:max_length]):
        if ch in charset:
            X[i] = charset[ch]
        else:
            X[i] = 0

    return X

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64
# ==================== 保持原有的序列编码函数 ====================
def smiles_sequence_embedding(sm, drug_vocab, seq_len=536):
    """改进版：在分词阶段同步记录原子位置"""
    content = []
    atom_indices = []
    flag = 0
    pos_offset = 1  # 因为要添加SOS标记，所以位置需要偏移

    # 原子符号检测函数
    def is_atom_symbol(s):
        return s in {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H',
                     'Na', 'Mg', 'K', 'Ca', 'Fe', 'Cu', 'Zn'}

    # 2-gram分词处理
    while flag < len(sm):
        # 尝试匹配2-gram
        if flag + 1 < len(sm):
            two_char = sm[flag:flag + 2]
            if two_char in drug_vocab.stoi:
                # 如果是原子符号（如Cl, Br）
                if is_atom_symbol(two_char):
                    atom_indices.append(len(content) + pos_offset)
                content.append(drug_vocab.stoi[two_char])
                flag += 2
                continue

        # 处理单字符
        one_char = sm[flag]
        if is_atom_symbol(one_char):
            atom_indices.append(len(content) + pos_offset)
        content.append(drug_vocab.stoi.get(one_char, drug_vocab.unk_index))
        flag += 1

    # 长度控制
    content = content[:seq_len]

    # 添加特殊标记
    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    content_len = len(content)

    # 调整原子位置（考虑填充）
    real_len = len(X)
    if seq_len > real_len:
        X.extend([drug_vocab.pad_index] * (seq_len - real_len))
    else:
        X = X[:seq_len]
        atom_indices = [i for i in atom_indices if i < seq_len]

    return torch.tensor(X), content_len, atom_indices

def smiles_to_onehot(smiles, max_length=100, charset_type='canonical'):
    """独立函数：SMILES转one-hot编码"""
    charset = CHARCANSMISET if charset_type == 'canonical' else CHARISOSMISET
    charset_size = CHARCANSMILEN if charset_type == 'canonical' else CHARISOSMILEN

    X = np.zeros((max_length, charset_size))

    for i, ch in enumerate(smiles[:max_length]):
        if ch in charset:
            X[i, (charset[ch] - 1)] = 1

    return X

def ligand_atom_to_feature(atom):
    #return [atom.get_atom_type(),
    #        atom.get_degree(),
    #        atom.get_Hydrogen_attached(),
    #        atom.is_aromatic(),]

    return [#atom_types[1] if atom.GetAtomicNum() not in atom_types else atom_types[atom.GetAtomicNum()],
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors = True),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),]

def ligand_to_graph(ligand):
    # atom
    atoms_feature_list = [ligand_atom_to_feature(atom) for atom in ligand.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype = np.float64)

    c = ligand.GetConformer()
    coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(ligand.GetNumAtoms())]
    node_positions = np.array(coordinates, dtype = np.float64)

    # bond
    if ligand.GetNumBonds() > 0:
        edge_list = []

        for bond in ligand.GetBonds():
            atom_u, atom_v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((atom_u, atom_v))
            edge_list.append((atom_v, atom_u))

        edge_index = np.array(edge_list, dtype = np.int64).T

    else:
        edge_index = np.empty((2, 0), dtype = np.int64)

    graph = {"node_feat": node_features,
            "num_nodes": node_features.shape[0],
            "edge_index": edge_index,
            "node_positions": node_positions}

    return graph


def protein_sequence_embedding(seq, target_vocab, tar_len=2600):
    """蛋白质序列嵌入编码"""
    content = []
    flag = 0
    # 2-gram分词处理
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)) and target_vocab.stoi.__contains__(seq[flag:flag + 2]):
            content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
            flag += 2
        else:
            content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
            flag += 1

    # 长度控制
    if len(content) > tar_len:
        content = content[:tar_len]

    # 添加特殊标记
    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    content_len = len(content)

    # 填充
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)

    return torch.tensor(X), content_len


if __name__ == '__main__':
    # 加载词汇表
    drug_vocab = WordVocab.load_vocab('../Vocab/smiles_vocab.pkl')
    target_vocab = WordVocab.load_vocab('../Vocab/protein_vocab.pkl')

    seq_len = 100     # 序列长度

    # 测试函数调用
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    encoded, seq_length, atom_positions = smiles_sequence_embedding(
        smiles, drug_vocab, seq_len
    )

    # 输出解析
    print(f"SMILES: {smiles}")
    print(f"编码后序列长度: {seq_length}")
    print(f"原子位置索引: {atom_positions}")
    print("编码序列（前20个）:", encoded[:20].tolist())