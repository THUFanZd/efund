import pandas as pd

def fill_na(df: pd.DataFrame):
    # 对所有列使用向下填充方法
    return df.bfill().ffill()

def normalize_column(df):
    """对DataFrame的每一列进行Z-score标准化"""
    return (df - df.mean()) / df.std()

def normalize_chart(df, nightlight_cols):
    """对DataFrame中指定的列使用统一的均值和方差进行Z-score标准化"""
    all_values = df[nightlight_cols].values.flatten()
    all_values = all_values[~pd.isna(all_values)]
    mean_val = all_values.mean()
    std_val = all_values.std()
    
    df[nightlight_cols] = (df[nightlight_cols] - mean_val) / std_val
    return df

if __name__ == '__main__':
    idx = input('输入保存结果文件的索引\n如果是1则结果无后缀, 最小是1\n')
    try:
        idx = int(idx)
    except ValueError:
        print('输入的索引不是整数')
        exit()
    assert idx >= 1, '索引不能小于1'

    df = pd.read_csv('../origin_merged_result.csv', encoding='gbk')
    l = []
    for i in range(len(df.columns)):
        if '夜光指数' in df.columns[i]:
            l.append(i)

    first_col = df.iloc[:, 0]  # date
    light_cols_idx = [df.columns[i] for i in l]
    fin_cols = [i for i in df.columns if i not in light_cols_idx]
    fin_cols = df[fin_cols]
    fin_cols = fill_na(fin_cols)
    light_cols = df[light_cols_idx]  # unnecessary
    light_cols = fill_na(light_cols)
    light_cols = normalize_chart(light_cols, light_cols_idx)

    # 合并处理后的列
    df = pd.concat([first_col, fin_cols, light_cols], axis=1)
    df.to_csv(f'../merged_result_{idx}.csv', index=False, encoding='gbk')
