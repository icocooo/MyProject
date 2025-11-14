import os

import dgl
import pandas as pd
import torch
import numpy as np

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DTIDataset(Dataset):
    def __init__(self,mode='train'):
        dir = '/home/icoco/PycharmProjects/MyProject/'
        protein_dir = dir + 'data/Davis/processed/pocket_graph/'
        compound_dir = dir + 'data/Davis/processed/compound_graph/'
        embedding_dir = dir + 'data/Davis/processed/ESM_embedding_pocket/'
        csv_path = dir + 'data/Davis/Davis.csv'
        if mode == 'train':
            csv_path = dir + 'data/Davis/train.csv'
        else:
            csv_path = dir + 'data/Davis/test.csv'
        self.df = pd.read_csv(csv_path, usecols=['COMPOUND_ID', 'PROTEIN_ID', 'REG_LABEL'],dtype='str')

        self.protein_graph_map = {}
        for id in os.listdir(protein_dir):
            id = id.removesuffix('.bin')
            protein_graph, _ = load_graphs(protein_dir + str(id) + '.bin')
            self.protein_graph_map[id] = protein_graph[0]

        self.compound_graph_map = {}
        for id in os.listdir(compound_dir):
            id = id.removesuffix('.bin')
            compound_graph, _ = load_graphs(compound_dir + str(id) + '.bin')
            self.compound_graph_map[id] = compound_graph[0]

        self.embedding_map = {}
        for id in os.listdir(embedding_dir):
            id = id.removesuffix('.npy')
            self.embedding_map[id] = np.load(embedding_dir + str(id) + '.npy', allow_pickle=True)
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        compound_graph_id = row['COMPOUND_ID']
        protein_graph_id = row['PROTEIN_ID']
        label = row['REG_LABEL']
        self.compound_graph = self.compound_graph_map[compound_graph_id]
        self.protein_graph = self.protein_graph_map[protein_graph_id]
        self.compound_graph = self.compound_graph_map[compound_graph_id]
        self.embedding = self.embedding_map[protein_graph_id]

        compound_len = self.compound_graph.num_nodes()
        protein_len = self.protein_graph.num_nodes()

        return self.compound_graph, self.protein_graph ,self.embedding,compound_len, protein_len, float(label)


    def collate(self, sample):
        batch_size = len(sample)

        compound_graph, protein_graph,protein_embedding, compound_len,protein_len, label = map(list, zip(*sample))
        max_protein_len = max(protein_len)

        for i in range(batch_size):
            if protein_embedding[i].shape[0] < max_protein_len:
                protein_embedding[i] = np.pad(protein_embedding[i], ((0, max_protein_len-protein_embedding[i].shape[0]), (0, 0)), mode='constant', constant_values = (0,0))

        compound_graph = dgl.batch(compound_graph).to(device)
        protein_graph = dgl.batch(protein_graph).to(device)
        label = torch.FloatTensor(label).to(device)
        protein_embedding = torch.FloatTensor(protein_embedding).to(device)

        return compound_graph, protein_graph,protein_embedding,label

