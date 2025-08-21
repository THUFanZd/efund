import os
import torch
import shutil
import pandas as pd
from data_find import find_macro_data, find_merged_data, find_conf_data

res_df = pd.read_excel('../待提交的预测结果及评价指标.xlsx')
target_date = res_df.iloc[:, 0]
target_date = target_date.dropna()

import json
with open('args.json', 'r') as f:
    all_args = json.load(f)

if __name__ == '__main__':
    # 同样处理 macro_hist_df 日期列
    arg = all_args[0]
    if arg.get('add_month', False):
        macro_hist_path = '../test_macro_hist_with_month.csv'
    else:
        macro_hist_path = '../test_macro_hist.csv'
    macro_hist_df = pd.read_csv(macro_hist_path, encoding='gbk')
    macro_stand = macro_hist_df.copy()
    macro_stand['日期'] = pd.to_datetime(macro_stand['日期'], errors='coerce')
    conf_df = pd.read_csv('../pbc_conference.csv')
    merged_df = pd.read_csv('../merged_result.csv', encoding='gbk')

    # 创建缓存路径
    cache_path = './cache_test'
    os.makedirs(cache_path, exist_ok=True)

    def format_date_for_lookup(date):
        return f"{date.year}/{date.month}/{date.day}"

    ex_args = {}
    for i, args in enumerate(all_args):
        if args["data_months"] == ex_args.get("data_months", None) and\
            args["num_months"] == ex_args.get("num_months", None):
            if not os.path.exists(f'{cache_path}/{i}'):
                shutil.copytree(f'{cache_path}/{i-1}', f'{cache_path}/{i}')
                print(f'copy {i-1} to {i}')

        else:
            ex_args = args
            os.makedirs(f'{cache_path}/{i}', exist_ok=True)
            for j, date in enumerate(target_date):
                idx = macro_stand[macro_stand['日期'] == date].index
                idx = idx[0]
                y = torch.tensor(macro_hist_df.iloc[idx][1:].values.astype(float), dtype=torch.float)
                
                def last_day_prev_month(date: pd.Timestamp, lag: int = 0) -> pd.Timestamp:
                    return date - pd.offsets.MonthEnd(lag)  # 推到前lag个月的最后一天

                date_str = format_date_for_lookup(last_day_prev_month(date, args.get("lag", 0)))
                macro_x = find_macro_data(macro_hist_df, date_str, args["num_months"])
                try:
                    merged_x = find_merged_data(merged_df, date_str, args["num_months"])
                except:
                    merged_x = find_merged_data(merged_df, "2025/8/31", args["num_months"])  # 最后有数据的月份
                conf_x = find_conf_data(conf_df, date_str, args["num_months"])

                torch.save((macro_x, merged_x, conf_x, y), f'{cache_path}/{i}/{j}.pt')
                print(f"Cached sample {i}/{len(all_args)} {j}/{len(target_date)} -> {date_str}")
