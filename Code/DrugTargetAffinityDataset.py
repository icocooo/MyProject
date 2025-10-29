import os

from torch.utils.data import Dataset
import csv

class DrugTargetAffinityDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.root = '../Data'
        self.file= 'davis.csv'
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.file)





