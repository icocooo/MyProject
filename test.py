import os
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
from dgl import load_graphs

from Code.DTIDataset import DTIDataset

dataset = DTIDataset()
print(dataset[0])