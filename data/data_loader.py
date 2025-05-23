import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.weekly_pattern_aligner import WeeklyPattern

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None, enable_data_cleaning=False, weekly_pattern_aligner : WeeklyPattern=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        #self.inverse = inverse
        self.enable_data_cleaning = enable_data_cleaning
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.weekly_pattern_aligner = weekly_pattern_aligner
        self.__read_data__()

    def __read_data__(self):
        # Determine file path based on enable_data_cleaning
        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path)

        dirty_cols = None
        if self.enable_data_cleaning:
            dirty_cols = df_raw['dirty'].values
            df_raw = df_raw.drop(columns=['dirty'])

        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        index = np.arange(len(df_data))

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.index = index[border1:border2]

        self.data_xs = []
        self.data_ys = []
        self.data_pxs = []
        self.data_pys = []
        self.length = 0
        for i in range(len(self.data_x) - self.in_len - self.out_len + 1):
            seq_x = self.data_x[i:i + self.in_len]
            seq_y = self.data_y[i + self.in_len:i + self.in_len + self.out_len]
            s_i, s_j, s_k = self.index[i], self.index[i+self.in_len], self.index[i + self.in_len + self.out_len - 1] + 1
            if self.enable_data_cleaning:
                if np.sum(dirty_cols[i:i + self.in_len + self.out_len]) > 0:
                    continue
            if self.weekly_pattern_aligner is not None:
                x_mean, x_std = seq_x.mean(axis=0), seq_x.std(axis=0)
                self.data_pxs.append(self.weekly_pattern_aligner.get_pattern(s_i, s_j) * x_std + x_mean)
                self.data_pys.append(self.weekly_pattern_aligner.get_pattern(s_j, s_k) * x_std + x_mean)
            self.data_xs.append(seq_x)
            self.data_ys.append(seq_y)
            self.length += 1
        print("Data length: ", self.length)

    def __getitem__(self, index):
        if self.weekly_pattern_aligner is not None:
            return self.data_xs[index], self.data_ys[index], self.data_pxs[index], self.data_pys[index]
        else:
            return self.data_xs[index], self.data_ys[index], None, None

    
    def __len__(self):
        return self.length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)