import os
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
from dgl import load_graphs
from torch.utils.data import DataLoader

from Code.DTIDataset import DTIDataset

dataset = DTIDataset()
compound_graph, protein_graph, embedding ,compound_len, protein_len, label = dataset[0]
# print(embedding.shape)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate, drop_last=True)
data = next(iter(train_loader))
print(data.shape)
# print(dataset[0])