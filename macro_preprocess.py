import numpy as np
import pandas as pd

def fill_na(df):
    # 对所有列使用向下填充方法
    return df.fillna(method='bfill').fillna(method='ffill')

if __name__ == '__main__':
    df = pd.read_csv('../origin_macro_hist.csv', encoding='gbk')
    df = fill_na(df)
    
    df.to_csv('../macro_hist.csv', index=False, encoding='gbk')
