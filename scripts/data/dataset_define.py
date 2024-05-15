import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class SingleTraitDataset(Dataset):
    def __init__(self, file_list: list, trait_name: str):
        self.xs, self.ys = None, None
        self.file_list = file_list
        self.trait_name = trait_name
        self.xs_use, self.ys_use = self.get_use_data()

    def __len__(self):
        return len(self.ys_use)

    def __getitem__(self, idx):
        return self.xs_use[idx], self.ys_use[idx]

    def _read_data(self):
        xs, ys = [], []
        for file in self.file_list:
            with open(file, 'rb') as f:
                sample_data = pickle.load(f)
                x = torch.tensor(sample_data['ref'].values[:-1])
                x = x.unsqueeze(0)
                y = sample_data[self.trait_name]
                xs.append(x)
                ys.append(y)
        return xs, ys

    def get_use_data(self):
        self.xs, self.ys = self._read_data()

        # clean nan value
        non_nan_indices = [i for i, item in enumerate(self.ys) if not math.isnan(item)]
        xs_use = [item for i, item in enumerate(self.xs) if i in non_nan_indices]
        ys_use = [item for i, item in enumerate(self.ys) if i in non_nan_indices]

        return xs_use, ys_use


class PretrainDataset(Dataset):
    def __init__(self, file_list: list, trait_name: str, scalers: dict={}):
        self.xs, self.ys = None, None
        self.file_list = file_list
        self.trait_name = trait_name
        self.scalers = scalers
        self.xs_use, self.ys_use = self.get_use_data()

    def __len__(self):
        return len(self.ys_use)

    def __getitem__(self, idx):
        return self.xs_use[idx], self.ys_use[idx]

    def _read_data(self):
        xs, ys = [], []
        for file in self.file_list:
            with open(file, 'rb') as f:
                sample_data = pickle.load(f)
                x = torch.tensor(sample_data['ref'].values[:-1])
                x = x.unsqueeze(0)
                y = sample_data[self.trait_name]
                xs.append(x)
                ys.append(y)
        return xs, ys

    def drop_nan(self):
        non_nan_indices = [i for i, item in enumerate(self.ys) if not math.isnan(item)]
        xs_nonnan = [item for i, item in enumerate(self.xs) if i in non_nan_indices]
        ys_nonnan = [item for i, item in enumerate(self.ys) if i in non_nan_indices]
        return xs_nonnan, ys_nonnan

    def normalize_ys(self, ys):
        ys_normed = ys.copy()

        if len(self.scalers) == 0:
            trait_values = np.array([])
            for y_dict in ys:
                trait_values = np.append(trait_values, y_dict) if not np.isnan(y_dict) else trait_values

            max_ = np.max(trait_values)
            min_ = np.min(trait_values)
            range_ = (max_ - min_)

            self.scalers = {'max': max_, 'min': min_, 'range': range_}

        else:
            max_ = self.scalers['max']
            min_ = self.scalers['min']
            range_ = self.scalers['range']

        for i, y_dict in enumerate(ys):
            ys_normed[i] = (y_dict - min_) / range_

        return ys_normed

    def get_use_data(self):
        self.xs, self.ys = self._read_data()
        xs, ys = self.drop_nan()
        xs_use = xs.copy()
        ys_use = self.normalize_ys(ys)
        return xs_use, ys_use


class MultiTraitsDataset(Dataset):
    def __init__(self, file_list: list, trait_names: list, scalers: dict={}):
        self.xs, self.ys = None, None
        self.file_list = file_list
        self.trait_names = trait_names
        self.scalers = scalers
        self.xs_use, self.ys_use = self.get_use_data()

    def __len__(self):
        return len(self.ys_use)

    def __getitem__(self, idx):
        return self.xs_use[idx], self.ys_use[idx]

    def _read_data(self):
        xs, ys = [], []
        for file in self.file_list:
            with open(file, 'rb') as f:
                sample_data = pickle.load(f)
                x = torch.tensor(sample_data['ref'].values[:-1])
                x = x.unsqueeze(0)

                y = {k: sample_data[k] for k in self.trait_names if k in sample_data.keys()}
                xs.append(x)
                ys.append(y)

        # xs: list(torch.tensor()), ys:list(dict：{str, torch.tensor(),...})
        return xs, ys

    def normalize_ys(self):
        ys_normed = self.ys.copy()

        for key in self.trait_names:
            if key not in self.scalers:
                trait_values = np.array([])
                for y_dict in self.ys:
                    trait_values = np.append(trait_values, y_dict[key]) if not np.isnan(y_dict[key]) else trait_values

                max_ = np.max(trait_values)
                min_ = np.min(trait_values)
                range_ = (max_ - min_)

                self.scalers[key] = {'max': max_, 'min': min_, 'range': range_}

            else:
                max_ = self.scalers[key]['max']
                min_ = self.scalers[key]['min']
                range_ = self.scalers[key]['range']

            for i, y_dict in enumerate(self.ys):
                ys_normed[i][key] = (y_dict[key] - min_) / range_

        return ys_normed

    def get_use_data(self):
        # else processing here
        self.xs, self.ys = self._read_data()
        xs_use = self.xs
        ys_use = self.normalize_ys()
        return xs_use, ys_use


if __name__ == '__main__':
    trait_names = ['LMA', 'EWT', 'RWC']
    with open('train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]

        dataset = MultiTraitsDataset(train_files, trait_names)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
        for data in dataloader:
            inp, lab = data
            lab_first = lab[trait_names[0]]
            print(inp.shape, lab_first.shape)
