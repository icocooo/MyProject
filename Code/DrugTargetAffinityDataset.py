import csv
import os
import pandas
from torch.utils.data import Dataset
import numpy as np

from Code.data_util1 import ligand_to_graph


class DrugTargetAffinityDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.root = '../Data'
        self.file= 'davis.csv'
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.file)
        all_data = csv.reader(path)
        x=np.array(all_data)
        print(x.shape)
path = os.path.join('../Data', 'davis.csv')

df = pandas.read_csv(path, index_col=0,header=0)
smiles = df.iloc[:50,0].values
targets = df.iloc[:50,1].values
affinity = df.iloc[:50,3].values
print(smiles[0],targets[0],affinity[0])






