import pandas as pd

macro_hist_df = pd.read_csv('../macro_hist.csv', encoding='gbk')
print('date: ', macro_hist_df.iloc[0][0])
l = macro_hist_df.iloc[0][1:]
print(l)
print(l.dtype)
print(l.values.astype(float))

num_month = 6
idx = macro_hist_df[macro_hist_df['日期'] == '2025/7/31'].index
a = macro_hist_df[idx[0]+1: idx[0]+num_month+1][1:]
print(a)
# 去掉a的第一列
a = a.drop(columns=['日期'])
print(a)
