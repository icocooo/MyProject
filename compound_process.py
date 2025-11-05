import dgl
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from dgl.data.utils import save_graphs

from scipy import sparse as sp
from itertools import permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs

import os

# 设置环境变量，避免libiomp5md.dll冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore")

# SMILES字符到整数的映射字典，用于序列化SMILES字符串
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    """
    将SMILES字符串转换为整数序列

    参数:
        line: SMILES字符串
        smi_ch_ind: 字符到整数的映射字典
        MAX_SMI_LEN: 最大序列长度，默认为100

    返回:
        X: 整数序列数组
    """
    # 创建全零数组用于存储编码结果
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    # 遍历SMILES字符串中的每个字符并进行编码
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    """
    对输入值进行one-hot编码，要求输入值必须在允许集合中

    参数:
        x: 需要编码的值
        allowable_set: 允许的值集合

    返回:
        one-hot编码列表
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """
    对输入值进行one-hot编码，如果输入值不在允许集合中，则映射到最后一个元素

    参数:
        x: 需要编码的值
        allowable_set: 允许的值集合

    返回:
        one-hot编码列表
    """
    if x not in allowable_set:
        x = allowable_set[-1]  # 映射到集合的最后一个元素
    return [x == s for s in allowable_set]


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    为图节点添加拉普拉斯位置编码

    参数:
        g: DGL图对象
        pos_enc_dim: 位置编码的维度

    返回:
        添加了位置编码的图对象
    """
    # 获取图的邻接矩阵（压缩稀疏行格式）
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    # 计算度矩阵的逆平方根（用于归一化）
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    # 计算归一化拉普拉斯矩阵
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # 使用numpy计算特征值和特征向量
    EigVal, EigVec = np.linalg.eig(L.toarray())
    # 对特征值进行排序（升序）
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 如果特征向量数量不足，进行填充
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)

    # 将位置编码添加到图节点数据中（跳过第一个特征向量）
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def atom_features(atom, explicit_H=False, use_chirality=True):
    """
    生成原子特征向量，包括原子符号、度、形式电荷等化学信息

    参数:
        atom: RDKit原子对象
        explicit_H: 是否显式处理氢原子
        use_chirality: 是否使用手性信息

    返回:
        原子特征向量
    """
    # 原子符号类型（17维）
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    # 原子度（7维）
    degree = [0, 1, 2, 3, 4, 5, 6]
    # 杂化类型（6维）
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']

    # 组合各种特征：符号(17) + 度(7) + 形式电荷(1) + 自由基电子数(1) + 杂化类型(6) + 芳香性(1) = 33维
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]

    # 如果不显式处理氢原子，添加氢原子数量信息（5维）
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 33+5=38

    # 如果使用手性信息，添加手性特征（3维）
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3=41

    return results


def bond_features(bond, use_chirality=True):
    """
    生成化学键特征向量

    参数:
        bond: RDKit键对象
        use_chirality: 是否使用手性信息

    返回:
        化学键特征向量
    """
    bt = bond.GetBondType()
    # 基础键特征：单键、双键、三键、芳香键、共轭、在环内
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

    # 如果使用手性信息，添加立体化学特征
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats).astype(int)


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    """
    将SMILES字符串转换为DGL图结构

    参数:
        smiles: SMILES字符串
        explicit_H: 是否显式处理氢原子
        use_chirality: 是否使用手性信息

    返回:
        DGL图对象
    """
    try:
        # 将SMILES字符串解析为RDKit分子对象
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")

    # 创建空的DGL图
    g = dgl.DGLGraph()

    # 添加节点（原子）
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # 为每个原子生成特征向量
    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])

    # 如果使用手性信息，添加手性中心特征
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    # 将原子特征添加到图节点数据中
    g.ndata["atom"] = torch.tensor(atom_feats)

    # 添加边（化学键）
    src_list = []  # 源节点列表
    dst_list = []  # 目标节点列表
    bond_feats_all = []  # 所有边的特征列表

    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()  # 键的起始原子索引
        v = bond.GetEndAtomIdx()  # 键的结束原子索引

        # 生成键特征
        bond_feats = bond_features(bond, use_chirality=use_chirality)

        # 添加双向边（无向图）
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)  # 每条边添加两次（双向）

    # 在图中添加边
    g.add_edges(src_list, dst_list)

    # 将键特征添加到图边数据中
    g.edata["bond"] = torch.tensor(np.array(bond_feats_all))

    # 添加拉普拉斯位置编码（8维）
    g = laplacian_positional_encoding(g, pos_enc_dim=8)
    return g


def Compound_graph_construction(id, compound_values, dir_output):
    """
    构建化合物图结构并保存

    参数:
        id: 化合物ID列表
        compound_values: 化合物SMILES字符串列表
        dir_output: 输出目录路径
    """
    N = len(compound_values)
    for no, data in enumerate(id):
        compounds_g = list()
        # 显示处理进度
        print('/'.join(map(str, [no + 1, N])))

        # 获取当前化合物的SMILES字符串
        smiles_data = compound_values[no]
        # 将SMILES转换为图结构
        compound_graph = smiles_to_graph(smiles_data)
        compounds_g.append(compound_graph)

        # 保存图结构到文件
        dgl.save_graphs(dir_output + '/compound_graph/' + str(data) + '.bin', list(compounds_g))


def Compound_graph_process(dataset, fold, dir_output, id_train, id_test):
    """
    处理训练集和测试集的化合物图数据

    参数:
        dataset: 数据集名称
        fold: 交叉验证的折数
        dir_output: 输出目录
        id_train: 训练集化合物ID列表
        id_test: 测试集化合物ID列表
    """
    compounds_graph_train, compounds_graph_test = [], []

    # 处理训练集化合物图
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        # 加载单个化合物的图数据
        compound_graph_train, _ = load_graphs('data/' + dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        compounds_graph_train.append(compound_graph_train[0])
    print(len(compounds_graph_train))
    # 保存训练集所有化合物的图数据
    dgl.save_graphs(dir_output + '/train/fold/' + str(fold) + '/compound_graph.bin', compounds_graph_train)

    # 处理测试集化合物图
    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        compound_graph_test, _ = load_graphs('data/' + dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        compounds_graph_test.append(compound_graph_test[0])
    print(len(compounds_graph_test))
    # 保存测试集所有化合物的图数据
    dgl.save_graphs(dir_output + '/test/fold/' + str(fold) + '/compound_graph.bin', compounds_graph_test)


def Compound_id_process(dataset, fold, dir_output, id_train, id_test):
    """
    保存训练集和测试集的化合物ID

    参数:
        dataset: 数据集名称
        fold: 交叉验证的折数
        dir_output: 输出目录
        id_train: 训练集化合物ID列表
        id_test: 测试集化合物ID列表
    """
    compounds_id_train, compounds_id_test = [], []

    # 保存训练集化合物ID
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        compounds_id_train.append(id)
    np.save(dir_output + '/train/fold/' + str(fold) + '/compound_id.npy', compounds_id_train)

    # 保存测试集化合物ID
    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        compounds_id_test.append(id)
    np.save(dir_output + '/test/fold/' + str(fold) + '/compound_id.npy', compounds_id_test)


def Label_process(dataset, fold, dir_output, label_train, label_test):
    """
    保存训练集和测试集的标签数据

    参数:
        dataset: 数据集名称
        fold: 交叉验证的折数
        dir_output: 输出目录
        label_train: 训练集标签列表
        label_test: 测试集标签列表
    """
    labels_train, labels_test = [], []

    # 保存训练集标签
    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(dir_output + '/train/fold/' + str(fold) + '/label.npy', labels_train)

    # 保存测试集标签
    N = len(label_test)
    for no, data in enumerate(label_test):
        print('/'.join(map(str, [no + 1, N])))
        labels_test.append(data)
    np.save(dir_output + '/test/fold/' + str(fold) + '/label.npy', labels_test)


if __name__ == '__main__':
    """
    主程序入口：执行化合物数据的完整预处理流程
    """
    # 数据集配置
    dataset = 'Davis'
    file_path = 'data/' + dataset + '/DTA/fold/'
    file_path_compound = 'data/' + dataset + '/' + dataset + '_compound_mapping.csv'
    dir_output = ('data/' + dataset + '/processed/')

    # 创建输出目录
    os.makedirs(dir_output, exist_ok=True)

    # 加载化合物映射数据
    raw_data_compound = pd.read_csv(file_path_compound)
    compound_values = raw_data_compound['COMPOUND_SMILES'].values
    compound_id_unique = raw_data_compound['COMPOUND_ID'].values

    N = len(compound_values)
    compound_max_len = 100  # SMILES字符串最大长度限制

    # 第一步：构建所有化合物的图结构
    Compound_graph_construction(id=compound_id_unique, compound_values=compound_values, dir_output=dir_output)

    # 第二步：对每个交叉验证折数进行处理
    for fold in range(1, 6):
        # 加载训练集和测试集数据
        train_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_train.csv')
        test_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_test.csv')

        # 提取化合物ID和标签
        compound_id_train = train_data['COMPOUND_ID'].values
        compound_id_test = test_data['COMPOUND_ID'].values
        label_train = train_data['REG_LABEL'].values
        label_test = test_data['REG_LABEL'].values

        # 处理化合物图数据
        Compound_graph_process(dataset=dataset, fold=fold, id_train=compound_id_train,
                               id_test=compound_id_test, dir_output=dir_output)
        # 处理化合物ID数据
        Compound_id_process(dataset=dataset, fold=fold, id_train=compound_id_train,
                            id_test=compound_id_test, dir_output=dir_output)
        # 处理标签数据
        Label_process(dataset=dataset, fold=fold, dir_output=dir_output,
                      label_train=label_train, label_test=label_test)

    # 完成预处理
    print('The preprocess of ' + dataset + ' dataset has finished!')