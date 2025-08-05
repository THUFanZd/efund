import os
import math
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class MyDataset(Dataset):
    def __init__(self, args, flag):
        super(MyDataset, self).__init__()
        self.args = args
        self.flag = flag
        self.cache_dir = f'{args["cache_path"]}/{flag}'
        self.file_list = sorted([
            os.path.join(self.cache_dir, fname)
            for fname in os.listdir(self.cache_dir)
            if fname.endswith('.pt')
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return torch.load(self.file_list[idx])

    def get_merge_dim(self):
        # 加载第一个样本取维度
        _, merged_x, _, _ = self.__getitem__(0)
        return merged_x.shape[-1]


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
        
        generator = torch.Generator()
        generator.manual_seed(42)
        sampler = WeightedRandomSampler(weights=sample_weights, 
                                        num_samples=args["sample_amount"],
                                        replacement=True,
                                        generator=generator)  # 为了可复现
        
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