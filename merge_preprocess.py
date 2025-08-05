import pandas as pd

def fill_na(df):
    # 对所有列使用向下填充方法
    return df.fillna(method='bfill').fillna(method='ffill')

def normalize(df):  # TODO 是否合理?
    """对DataFrame的每一列进行Z-score标准化"""
    return (df - df.mean()) / df.std()

if __name__ == '__main__':
    df = pd.read_csv('../origin_merged_result.csv', encoding='gbk')
    # 分离第一列和其他列
    first_col = df.iloc[:, 0]
    other_cols = df.iloc[:, 1:]
    # 对其他列进行填充和标准化
    other_cols = fill_na(other_cols)
    other_cols = normalize(other_cols)
    # 合并处理后的列
    df = pd.concat([first_col, other_cols], axis=1)
    df.to_csv('../merged_result.csv', index=False, encoding='gbk')