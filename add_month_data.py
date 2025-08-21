import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../macro_hist.csv', encoding='gbk')
    df['月份'] = df['日期'].apply(lambda x: x.split('/')[1])
    df.to_csv('../macro_hist_with_month.csv', index=False, encoding='gbk')

