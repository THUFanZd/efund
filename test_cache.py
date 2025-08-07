import os
import torch
import pandas as pd
from data_find import find_macro_data, find_merged_data, find_conf_data


res_df = pd.read_excel('../待提交的预测结果及评价指标.xlsx')
target_date = res_df.iloc[:, 0]
target_date = target_date.dropna()


import json
with open('args.json', 'r') as f:
    args = json.load(f)
    assert type(args) == dict

cache_path='./cache_test'


if __name__ == '__main__':
    # 同样处理 macro_hist_df 日期列
    macro_hist_df = pd.read_csv('../test_macro_hist.csv', encoding='gbk')
    macro_stand = macro_hist_df.copy()
    macro_stand['日期'] = pd.to_datetime(macro_stand['日期'], errors='coerce')
    conf_df = pd.read_csv('../pbc_conference.csv')
    merged_df = pd.read_csv('../merged_result.csv', encoding='gbk')

    # 创建缓存路径
    cache_path = './cache_test'
    os.makedirs(cache_path, exist_ok=True)

    def format_date_for_lookup(date):
        return f"{date.year}/{date.month}/{date.day}"

    # 遍历处理
    for i, date in enumerate(target_date):
        # 在 macro_hist_df 中找到该日期对应的行
        idx = macro_stand[macro_stand['日期'] == date].index

        if len(idx) == 0:
            # print(f"日期 {date} 不存在于 macro_hist_df 中，跳过")
            # continue
            # 将y设置为全0向量即可
            y = torch.zeros(args['macro_dim'], dtype=torch.float)
        else:
            idx = idx[0]
            y = torch.tensor(macro_hist_df.iloc[idx][1:].values.astype(float), dtype=torch.float)

        date_str = format_date_for_lookup(date)
        macro_x = find_macro_data(macro_hist_df, date_str, args["num_months"])
        try:
            merged_x = find_merged_data(merged_df, date_str, args["num_months"])
        except:
            merged_x = find_merged_data(merged_df, "2025/8/31", args["num_months"])
        conf_x = find_conf_data(conf_df, date_str, args["num_months"])

        torch.save((macro_x, merged_x, conf_x, y), f'{cache_path}/{i}.pt')
        print(f"Cached sample {i}/{len(target_date)} -> {date.strftime('%Y/%m/%d')}")
