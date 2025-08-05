import os
import torch
import pandas as pd
from data_find import find_macro_data, find_merged_data, find_conf_data

def preprocess_and_save(args, split="train"):
    os.makedirs(f'./cache/{split}', exist_ok=True)

    macro_hist_df = pd.read_csv('../macro_hist.csv', encoding='gbk')[4:args["data_months"]+4+args["num_months"]+1]
    conf_df = pd.read_csv('../pbc_conference.csv')
    merged_df = pd.read_csv('../merged_result.csv', encoding='gbk')

    total = len(macro_hist_df)
    test_size = int(total * args["test_size"])
    if split == "train":
        indices = range(test_size, total)
    else:  # test
        indices = range(test_size)

    for i, idx in enumerate(indices):
        date_str = macro_hist_df.iloc[idx][0]
        y = torch.tensor(macro_hist_df.iloc[idx][1:].values.astype(float), dtype=torch.float)
        macro_x = find_macro_data(macro_hist_df, date_str, args["num_months"])
        merged_x = find_merged_data(merged_df, date_str, args["num_months"])
        conf_x = find_conf_data(conf_df, date_str, args["num_months"])
        torch.save((macro_x, merged_x, conf_x, y), f'./cache/{split}/{i}.pt')
        print(f"Cached sample {i}/{len(indices)} -> {date_str}")

if __name__ == '__main__':
    import json
    with open('args.json', 'r') as f:
        args = json.load(f)

    preprocess_and_save(args, split='train')
    preprocess_and_save(args, split='test')
