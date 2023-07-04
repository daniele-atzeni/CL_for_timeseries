from __future__ import print_function, division
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data import Dataset

# Using the sampler from https://github.com/ufoym/imbalanced-dataset-sampler
from torch.utils.data.sampler import Sampler


class LOB_WF(Dataset):

    def __init__(self, h5_path, split=1,
                 train=True, n_window=1, normalization=None, epsilon=1e-15, horizon=0):
        """
        Loads the LoB dataset and prepares it to perform an anchored walk-forward evaluation
        :param h5_path:  Path to the h5 file containing the dataset
        :param split: split to use (0 to 8)
        :param train: whether to load the train or the test split
        :param n_window: window of features before the current time stamp to load
        :param normalization: None or 'std' (z-score normalization)
        :param epsilon: epsilon to be used to ensure the stability of the normalization
        :param horizon: the prediction horizon (0 -> next (10), 1-> next 5 (50), 2-> next 10 (100))
        """

        self.window = n_window

        assert 0 <= split <= 8
        assert 0 <= horizon <= 2

        # Translate the prediction to horizon to the horizon (as encoded in the data)
        if horizon == 1:
            horizon = 3
        elif horizon == 2:
            horizon = 4

        # Load the data
        file = h5py.File(h5_path, 'r', )    ####### file is a sort of dictionary where each value is a HDF5 dataset, treatable as a np matrix
        features = np.array(file['features']).astype(float)                             # dataset shape (453975, 144) type f4
        targets = np.array(file['targets']).astype(int)                                 # dataset shape (453975, 5) type i4
        day_train_split_idx = np.array(file['day_train_split_idx']).astype('bool')      # dataset shape (9, 453975) type |b1            NB 9 is the max number of splits
        day_test_split_idx = np.array(file['day_test_split_idx']).astype('bool')        # dataset shape (9, 453975) type |b1
        stock_idx = np.array(file['stock_idx']).astype('bool')                          # dataset shape (5, 453975) type |b1
        file.close()

        ###############
        # 9 is the number of splits created
        # 5 is the number of companies
        # 144 is the number of features

        # basically they use day 1 for training and day 2 for evaluation, then day 1 and 2 for training and day 3 for evaluation etc
        # this process is repeated 9 times, obtaining the number of splits. Indeed the number of element in the train at i+1 is 
        # equal to the number of train at i + number of test at i 

        # each row of targets represents the labels for each of the five company, which are in the set {0, 1, 2}

        # each column of stock_idx represents the stock to which the current row is about

        # each column of train/test idxs is a boolean vector with True if the current row is in a split
        ###############

        # Get the data for the specific split and setup (train/test)
        if train:
            idx = day_train_split_idx[split]

            # Get the statistics needed for normalization
            if normalization != None:
                self.mean = np.mean(features[idx], axis=0)
                self.std = np.std(features[idx], axis=0)
                features = (features - self.mean) / (self.std + epsilon)
        else:
            idx = day_test_split_idx[split]

            # Also get the train data to normalize the test data accordingly (if needed)
            if normalization != None:
                train_idx = day_train_split_idx[split]
                self.mean = np.mean(features[train_idx], axis=0)
                self.std = np.std(features[train_idx], axis=0)
                features = (features - self.mean) / (self.std + epsilon)
                del train_idx

        # Get the data per stock
        self.features_per_stock = []
        self.labels = []
        for i in range(len(stock_idx)):         # same as range(stock_idx.shape[1])
            cur_idx = np.logical_and(idx, stock_idx[i])
            self.features_per_stock.append(features[cur_idx])
            self.labels.append(targets[cur_idx, horizon])   # why horizon?? targets is n_record x n_stock   (453975, 5)

        # Create a lookup table to find the correct stock
        self.look_up_margins = []
        current_sum = 0
        for i in range(len(self.features_per_stock)):
            # Remove n_window since they are used to ensure that we are always operate on a full window
            cur_limit = self.features_per_stock[i].shape[0] - n_window - 1
            current_sum += cur_limit
            self.look_up_margins.append(current_sum)

        # Get the total number of samples
        self.n = self.look_up_margins[-1]
        self.n_stocks = len(self.look_up_margins)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Get the stock id
        stock_id = self.n_stocks - 1
        for i in range(self.n_stocks - 1):
            if idx < self.look_up_margins[i]:
                stock_id = i
                break

        # Get the in-split idx for the corresponding stock
        if stock_id > 0:
            in_split_idx = idx - self.look_up_margins[stock_id - 1]
        else:
            in_split_idx = idx

        # Get the actual data
        cur_idx = in_split_idx + self.window
        data = self.features_per_stock[stock_id][cur_idx - self.window:cur_idx]
        label = self.labels[stock_id][cur_idx]
        return torch.from_numpy(data), torch.from_numpy(np.array([label]))




class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
        Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        Using the sampler from https://github.com/ufoym/imbalanced-dataset-sampler
    """

    def __init__(self, dataset):

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        data, label = dataset[idx]
        return int(label[0])

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples