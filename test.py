import os
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
from dgl import load_graphs
id1 = 'A0A2K5TT62'
id2 = '44259'
dataset = 'Davis'
file_path = 'data/' + dataset + '/DTA/fold/'
protein_df = pd.read_csv('Code/data/Davis/Davis_protein_mapping.csv')
protein_x = np.array_split(protein_df,5)
protein_df = pd.read_csv('Code/data/Davis/Davis_compound_mapping.csv')
compound_x = np.array_split(protein_df,5)
x , _= load_graphs('data/' + dataset + '/processed' + '/protein_graph/' + str(id1) + '.bin')
y , _= load_graphs('data/' + dataset + '/processed' + '/compound_graph/' + str(id2) + '.bin')
print(x)
print(y)