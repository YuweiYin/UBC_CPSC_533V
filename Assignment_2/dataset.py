import pickle
import torch
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # TODO YOUR CODE HERE
        # raise NotImplementedError()

        return {
            "state": torch.tensor(item[0], dtype=torch.float),
            "action": torch.tensor(item[1], dtype=torch.long)
        }
