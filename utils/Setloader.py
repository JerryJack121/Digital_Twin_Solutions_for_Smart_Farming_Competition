import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Setloader(Dataset):
    def __init__(self, data, label):
        super(Setloader, self).__init__()
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

class TestSetloader(Dataset):
    def __init__(self, data):
        super(TestSetloader, self).__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]