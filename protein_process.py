import os
import pickle
import timeit

import deepchem
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from scipy import sparse as sp
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import warnings

warnings.filterwarnings("ignore")

# 设置环境变量，避免libiomp5md.dll冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置设备为GPU
device = torch.device('cpu')

# 金属元素列表，用于特殊残基识别
METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]

# 残基最大原子数限制
RES_MAX_NATOMS = 24

# 氨基酸字符到整数的映射字典
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    """
    将蛋白质序列转换为整数编码序列

    参数:
        line: 蛋白质氨基酸序列字符串
        smi_ch_ind: 字符到整数的映射字典
        MAX_SEQ_LEN: 最大序列长度限制

    返回:
        X: 整数编码序列数组
    """
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    """
    对输入值进行严格的one-hot编码

    参数:
        x: 需要编码的值
        allowable_set: 允许的值集合

    返回:
        one-hot编码列表
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """
    对输入值进行one-hot编码，未知值映射到集合最后一个元素

    参数:
        x: 需要编码的值
        allowable_set: 允许的值集合

    返回:
        one-hot编码列表
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    """
    计算残基内部原子距离统计特征

    参数:
        res: MDAnalysis残基对象

    返回:
        距离统计特征列表
    """
    try:
        # 选择残基内所有原子
        xx = res.atoms
        # 计算原子间自距离矩阵
        dists = distances.self_distance_array(xx.positions)
        # 选择特定原子类型
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        # 返回归一化的距离特征
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    """
    计算残基的二面角特征

    参数:
        res: MDAnalysis残基对象

    返回:
        二面角特征列表（归一化）
    """
    try:
        # 计算phi二面角
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        # 计算psi二面角
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        # 计算omega二面角
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        # 计算chi1二面角
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        # 返回归一化的二面角值
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    """
    计算残基的综合特征向量

    参数:
        res: MDAnalysis残基对象

    返回:
        残基特征向量数组
    """
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32维：残基类型
                    obtain_self_dist(res) +  # 5维：距离统计特征
                    obtain_dihediral_angles(res)  # 4维：二面角特征
                    )


def obtain_resname(res):
    """
    获取标准化的残基名称

    参数:
        res: MDAnalysis残基对象

    返回:
        标准化的残基名称
    """
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    # 如果是金属元素，标记为"M"
    if resname in METAL:
        return "M"
    else:
        return resname


def obatin_edge(u, cutoff=10.0):
    """
    构建残基间的边连接关系

    参数:
        u: MDAnalysis Universe对象
        cutoff: 距离截断值（Å）

    返回:
        edgeids: 边连接索引列表
        distm: 距离特征矩阵
    """
    edgeids = []
    dismin = []
    dismax = []
    # 遍历所有残基对
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        # 如果最小距离小于截断值，则建立连接
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)  # 归一化
            dismax.append(dist.max() * 0.1)  # 归一化
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    """
    检查两个残基是否通过肽键连接

    参数:
        u: MDAnalysis Universe对象
        i, j: 残基索引

    返回:
        1: 连接, 0: 不连接
    """
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        # 检查键连接关系
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):
    """
    计算两个残基间原子距离矩阵

    参数:
        res1, res2: MDAnalysis残基对象

    返回:
        原子距离矩阵
    """
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def load_protein(protpath, explicit_H=False, use_chirality=True):
    """
    从PDB文件加载蛋白质分子

    参数:
        protpath: PDB文件路径
        explicit_H: 是否显式处理氢原子
        use_chirality: 是否使用手性信息

    返回:
        RDKit分子对象
    """
    mol = Chem.MolFromPDBFile(protpath, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def prot_to_graph(id, prot_pdb, cutoff=10.0):
    """
    将蛋白质转换为DGL图结构，聚焦于结合口袋

    参数:
        id: 蛋白质ID
        prot_pdb: PDB文件路径
        cutoff: 距离截断值

    返回:
        口袋子图的DGL图对象
    """
    # 使用DeepChem的凸包口袋查找器
    pk = deepchem.dock.ConvexHullPocketFinder()
    # 加载蛋白质结构
    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    Chem.AssignStereochemistryFrom3D(prot)
    # 创建MDAnalysis Universe对象
    u = mda.Universe(prot)
    g = dgl.DGLGraph()

    # 添加节点（残基）
    num_residues = len(u.residues)
    g.add_nodes(num_residues)

    # 计算残基特征
    res_feats = np.array([calc_res_features(res) for res in u.residues])

    # 加载ESM预训练嵌入
    esm_feats = np.load('data/Davis/processed/ESM_embedding_pocket/' + id + '.npy', allow_pickle=True)
    len_esm = np.size(esm_feats, 0)
    # 对齐ESM嵌入维度
    if len_esm < num_residues:
        esm_feats = np.pad(esm_feats, ((0, num_residues - len_esm), (0, 0)), 'constant')
    elif len_esm > num_residues:
        esm_feats = esm_feats[:num_residues, :]

    # 合并残基特征和ESM嵌入
    prot_feats = np.concatenate((res_feats, esm_feats), axis=1)
    g.ndata["feats"] = torch.tensor(prot_feats)

    # 构建边连接
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)
    g.add_edges(src_list, dst_list)

    # 添加位置信息
    g.ndata["ca_pos"] = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    g.ndata["center_pos"] = torch.tensor(u.atoms.center_of_mass(compound='residues'))

    # 计算距离矩阵特征
    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    cadist = torch.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    cedist = torch.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1

    # 检查连接性
    edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))

    # 设置边特征
    g.edata["feats"] = torch.cat(
        [edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm)], dim=1)

    # 清理临时位置数据
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")

    # 获取CA原子位置用于口袋检测
    ca_pos = np.array(np.array([obtain_ca_pos(res) for res in u.residues]))

    # 查找结合口袋
    pockets = pk.find_pockets(prot_pdb)
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        idxs = []
        # 选择位于口袋内的残基
        for idx in range(ca_pos.shape[0]):
            if x_min < ca_pos[idx][0] < x_max and y_min < ca_pos[idx][1] < y_max and z_min < ca_pos[idx][2] < z_max:
                idxs.append(idx)

    # 创建口袋子图
    g_pocket = dgl.node_subgraph(g, idxs)
    # 添加拉普拉斯位置编码
    g_pocket = laplacian_positional_encoding(g_pocket, pos_enc_dim=8)
    return g_pocket


def obtain_ca_pos(res):
    """
    获取残基的CA原子位置

    参数:
        res: MDAnalysis残基对象

    返回:
        CA原子位置或残基中心位置
    """
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]  # 金属原子取第一个原子位置
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]  # 获取CA原子位置
            return pos
        except:  # 如果缺少CA原子，使用残基中心位置
            return res.atoms.positions.mean(axis=0)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    为图节点添加拉普拉斯位置编码

    参数:
        g: DGL图对象
        pos_enc_dim: 位置编码维度

    返回:
        添加位置编码后的图对象
    """
    # 计算归一化拉普拉斯矩阵
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # 特征值分解
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # 按特征值排序
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 维度对齐处理
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)

    # 添加位置编码（跳过第一个常数特征向量）
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def Protein_graph_construction(id, dir_output):
    """
    构建蛋白质图结构并保存

    参数:
        id: 蛋白质ID列表
        dir_output: 输出目录
    """
    N = len(id)
    start = timeit.default_timer()
    for no, data in enumerate(id):
        proteins_g = list()
        print('/'.join(map(str, [no + 1, N])))
        pdb_pdb = 'data/' + dataset + '/PDB_AF2/' + data + '.pdb'
        if not os.path.exists(dir_output + '/protein_graph/' + data + '.bin'):
            print('保存',data)
            # 转换为图结构
            protein_graph = prot_to_graph(data, pdb_pdb, cutoff=10.0)
            proteins_g.append(protein_graph)
            # 保存图数据
            dgl.save_graphs(dir_output + '/protein_graph/' + data + '.bin', list(proteins_g))

        end = timeit.default_timer()
        time = end - start
        print(round(time, 2))


def Protein_graph_process(dataset, fold, dir_output, id_train, id_test):
    """
    处理训练集和测试集的蛋白质图数据

    参数:
        dataset: 数据集名称
        fold: 交叉验证折数
        dir_output: 输出目录
        id_train: 训练集ID列表
        id_test: 测试集ID列表
    """
    proteins_graph_train, proteins_graph_test = [], []

    # 处理训练集
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        protein_graph_train, _ = load_graphs('data/' + dataset + '/processed' + '/pocket_graph/' + str(id) + '.bin')
        proteins_graph_train.append(protein_graph_train[0])
    print(len(proteins_graph_train))
    dgl.save_graphs(dir_output + '/train/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_train)

    # 处理测试集
    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        protein_graph_test, _ = load_graphs('data/' + dataset + '/processed' + '/pocket_graph/' + str(id) + '.bin')
        proteins_graph_test.append(protein_graph_test[0])
    print(len(proteins_graph_test))
    dgl.save_graphs(dir_output + '/test/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_test)


def Protein_embedding_process(dataset, fold, dir_output, id_train, id_test):
    """
    处理训练集和测试集的蛋白质嵌入数据

    参数:
        dataset: 数据集名称
        fold: 交叉验证折数
        dir_output: 输出目录
        id_train: 训练集ID列表
        id_test: 测试集ID列表
    """
    proteins_embedding_train, proteins_embedding_test = [], []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        protein_embedding_train = np.load(
            'data/' + dataset + '/processed' + '/ESM_embedding_pocket/' + str(id) + '.npy', allow_pickle=True)
        proteins_embedding_train.append(protein_embedding_train)
    print(len(proteins_embedding_train))
    np.save(dir_output + '/train/fold/' + str(fold) + '/protein_embedding.npy', proteins_embedding_train)

    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        protein_embedding_test = np.load('data/' + dataset + '/processed' + '/ESM_embedding_pocket/' + str(id) + '.npy',
                                         allow_pickle=True)
        proteins_embedding_test.append(protein_embedding_test)
    print(len(proteins_embedding_test))
    np.save(dir_output + '/test/fold/' + str(fold) + '/protein_embedding.npy', proteins_embedding_test)


def Protein_id_process(dataset, fold, dir_output, id_train, id_test):
    """
    处理训练集和测试集的蛋白质ID数据

    参数:
        dataset: 数据集名称
        fold: 交叉验证折数
        dir_output: 输出目录
        id_train: 训练集ID列表
        id_test: 测试集ID列表
    """
    proteins_id_train, proteins_id_test = [], []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        proteins_id_train.append(id)
    np.save(dir_output + '/train/fold/' + str(fold) + '/protein_id.npy', proteins_id_train)

    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        proteins_id_test.append(id)
    np.save(dir_output + '/test/fold/' + str(fold) + '/protein_id.npy', proteins_id_test)


if __name__ == '__main__':
    """
    主程序：执行蛋白质数据的完整预处理流程
    """
    # 数据集配置
    dataset = 'Davis'
    file_path = 'data/' + dataset + '/DTA/fold/'
    file_path_protein = 'data/' + dataset + '/' + dataset + '_protein_mapping.csv'
    dir_output = ('data/' + dataset + '/processed/')
    os.makedirs(dir_output, exist_ok=True)

    # 加载蛋白质映射数据
    raw_data_protein = pd.read_csv(file_path_protein)
    protein_id_unique = raw_data_protein['PROTEIN_ID'].values

    N = len(protein_id_unique)

    # 第一步：构建所有蛋白质的图结构
    Protein_graph_construction(id=protein_id_unique, dir_output=dir_output)

    # 第二步：对每个交叉验证折数进行处理
    for fold in range(1, 6):
        # 加载训练集和测试集数据
        train_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_train.csv')
        test_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_test.csv')

        # 提取蛋白质ID
        protein_id_train = train_data['PROTEIN_ID'].values
        protein_id_test = test_data['PROTEIN_ID'].values
 
        # 处理蛋白质图数据
        Protein_graph_process(dataset=dataset, fold=fold, id_train=protein_id_train, id_test=protein_id_test,
                              dir_output=dir_output)
        # 处理蛋白质嵌入数据
        Protein_embedding_process(dataset=dataset, fold=fold, id_train=protein_id_train, id_test=protein_id_test,
                                  dir_output=dir_output)
        # 处理蛋白质ID数据
        Protein_id_process(dataset=dataset, fold=fold, id_train=protein_id_train, id_test=protein_id_test,
                           dir_output=dir_output)

    # 完成预处理
    print('The preprocess of ' + dataset + ' dataset has finished!')