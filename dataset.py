import numpy as np
import pandas as pd
import random


class SplitedDataset:
    def __init__(self, root, standard=False):
        self.dataset = pd.read_csv(root, sep="\t").to_numpy()
        train_percent = 0.7
        train_num = int(len(self.dataset) * 0.7)
        trainset, testset = self.dataset[:train_num], self.dataset[train_num:]
        self.train_data, self.train_label = trainset[:, :-1],  trainset[:, -1]
        self.test_data, self.test_label = testset[:, :-1], testset[:, -1]
        if standard: 
            train_data_mean = self.train_data.mean(axis=0)
            train_data_std = np.std(self.train_data,axis=0)
            
            test_data_mean = self.test_data.mean(axis=0)
            test_data_std = np.std(self.test_data,axis=0)
            
            self.train_data = (self.train_data - train_data_mean) / train_data_std
            self.test_data = (self.test_data - test_data_mean) / test_data_std
            
    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle=False):
        if not shuffle:
            return self.train_data, self.train_label, self.test_data, self.test_label
        else:
            index = list(range(len(self.train_data))) 
            random.shuffle(index)
            return self.train_data[index], self.train_label[index], self.test_data, self.test_label

