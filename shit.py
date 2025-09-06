import pandas as pd

df = pd.read_csv("../merged_result_3.csv", encoding='gbk')
print("原始数据类型:")
print(df.dtypes)

# 查看前几行数据，检查是否有异常值
print("\n前5行数据:")
print(df.head())

# 检查每列中非数字值的数量
print("\n每列中非数字值的数量:")
for col in df.columns[1:]:
    non_numeric_count = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').isna().sum()
    print(f"{col}: {non_numeric_count}")

# 执行转换
try:
    # df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='raise'))
    # df.iloc[:, 1:] = df.iloc[:, 1:].astype('float')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    df = df.astype({col: 'float' for col in df.columns[1:]})
    print("\n转换后的数据类型:")
    print(df.dtypes)
except Exception as e:
    print(f"\n转换过程中出现错误: {e}")
    # 显示无法转换的值
    for col in df.columns[1:]:
        invalid_values = df[col][pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').isna()]
        if not invalid_values.empty:
            print(f"\n列 '{col}' 中无法转换的值:")
            print(invalid_values.unique())


exit()
import pandas as pd
df = pd.read_csv("../merged_result_3.csv", encoding='gbk')
# 除了第一列，其余列均转换为数字
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='raise'))
print(df.dtypes)