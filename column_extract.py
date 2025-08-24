import pandas as pd

with open('./test_args.json', 'r') as f:
    import json
    test_args = json.load(f)

csv_name = f"test_results_{test_args["test_model_name"]}"
csv_path = f'./test_res/{csv_name}.csv'
df = pd.read_csv(csv_path)
# 提取列名以pred开头的列并保留date列
pred_cols = df.filter(like='pred')
pred_cols['date'] = df['date']
# 将date列挪到第一列
pred_cols = pred_cols[['date'] + [col for col in pred_cols if col != 'date']]
print(pred_cols)
# 将提取的列存储为新的csv文件
pred_cols.to_csv(f'./test_res/{csv_name}_pred.csv', index=False)
pred_cols = df.filter(regex='mae|dir')
# 将mae开头的列放到一块，dir开头的列放到一块
mae_cols = pred_cols.filter(like='mae')
dir_cols = pred_cols.filter(like='dir')
pred_cols = pd.concat([mae_cols, dir_cols], axis=1)
pred_cols['date'] = df['date']
# 将date列挪到第一列
pred_cols = pred_cols[['date'] + [col for col in pred_cols if col != 'date']]
print(pred_cols)
# 将提取的列存储为新的csv文件
pred_cols.to_csv(f'./test_res/{csv_name}_eval.csv', index=False)
