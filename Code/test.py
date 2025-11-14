import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
import numpy as np
import pandas
import pandas as pd
import torch
from dgl import load_graphs
from torch.utils.data import DataLoader

from Code import gt_net_compound, gt_net_protein
from Code.DTIDataset import DTIDataset
import matplotlib.pyplot as plt

dir = './data/Davis'
df = pd.read_csv('./data/Davis/Davis.csv')
n = len(df) // 5

df_test = df.head(n)
df_train = df.iloc[n:]

df_test.to_csv( os.path.join(dir, 'test.csv'))
df_train.to_csv( os.path.join(dir, 'train.csv'))

