import math
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data_find import *


class MyDataset(Dataset):
    def __init__(self, args, flag="train"):
        super(MyDataset, self).__init__()
        # TODO 补充指数数据
        # TODO 数据正则化
        self.macro_hist_df = pd.read_csv('../macro_hist.csv', encoding='gbk')[4:args["data_months"]+4+args["num_months"]+1]
        self.conf_df = pd.read_csv('../pbc_conference.csv')
        self.merged_df = pd.read_csv('../merged_result.csv', encoding='gbk')
        self.test_size = args["test_size"]
        self.flag = flag
        self.args = args
        # self.merged_df.to_csv('../merged_result_tmp.csv', encoding='gbk', index=False)

    def __len__(self):
        if self.flag == "train":
            return self.args["data_months"] - int(self.args["data_months"] * self.test_size)
        elif self.flag == "test":
            return int(self.args["data_months"] * self.test_size)
    
    def __getitem__(self, idx):
        if self.flag == "train":
            idx = idx + int(self.args["data_months"] * self.args["test_size"])
        date_str = self.macro_hist_df.iloc[idx][0]
        y = self.macro_hist_df.iloc[idx][1:].values.astype(float)
        y = torch.tensor(y, dtype=torch.float)
        macro_x = find_macro_data(self.macro_hist_df, date_str, self.args["num_months"])
        merged_x = find_merged_data(self.merged_df, date_str, self.args["num_months"])
        conf_x = find_conf_data(self.conf_df, date_str, self.args["num_months"])
        return macro_x, merged_x, conf_x, y
    
    def get_merge_dim(self):
        return self.merged_df.shape[1] - 1


def generate_weights(length, max_weight=1.0, min_weight=0.1, decay_rate=0.1, method='linear'):
    if length == 1:
        return [max_weight]
    
    if method == 'linear':
        return [
            max_weight - (max_weight - min_weight) * (i / (length - 1))
            for i in range(length)
        ]

    elif method == 'exp':
        return [
            max_weight * math.exp(-decay_rate * (i / (length - 1)))
            for i in range(length)
        ]
    else:
        raise ValueError("method must be 'linear' or 'exp'")


def load_data(args, flag):
    data_set = MyDataset(
        args=args,
        flag=flag
    )

    if flag == "train":
        sample_weights = generate_weights(len(data_set), max_weight=1, min_weight=1 / args["max_min_weight_ratio"], \
                                          decay_rate=args["decay_rate"], method=args["weight_method"])
        sampler = WeightedRandomSampler(weights=sample_weights, 
                                        num_samples=args["sample_amount"],
                                        replacement=True)

        data_loader = DataLoader(
            data_set,
            batch_size=args["batch_size"],
            num_workers=0,
            drop_last=False,
            sampler=sampler)
        
    elif flag == "test":
        data_loader = DataLoader(
            data_set,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False)
        
    return data_set, data_loader