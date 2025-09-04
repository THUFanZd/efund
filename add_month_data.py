import pandas as pd

if __name__ == '__main__':
    # df = pd.read_csv('../test_macro_hist_with_month.csv', encoding='gbk')
    # print(df.head())
    # # 将日期列的分隔符'-'变成'/'
    # df['日期'] = df['日期'].apply(lambda x: x.replace('-', '/'))
    # df.to_csv('../test_macro_hist_with_month.csv', index=False, encoding='gbk')
    # print(df.head())

    # df = pd.read_csv('../macro_hist_with_month.csv', encoding='gbk')
    # print(df.head())



    # exit()
    df = pd.read_csv('../test_macro_hist.csv', encoding='gbk')
    df['月份'] = df['日期'].apply(lambda x: x.split('-')[1])
    df.to_csv('../test_macro_hist_with_month.csv', index=False, encoding='gbk')
    df = pd.read_csv('../macro_hist.csv', encoding='gbk')
    df['月份'] = df['日期'].apply(lambda x: x.split('/')[1])
    df.to_csv('../macro_hist_with_month.csv', index=False, encoding='gbk')
