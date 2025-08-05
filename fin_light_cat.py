import pandas as pd

shared_path = r"C:\Users\lzx\Desktop\大四暑\易方达杯"
# 读取两个表格（支持 .csv 或 .xlsx）
df1 = pd.read_excel(shared_path + '\\夜光卫星遥感数据.xlsx')  # 或 pd.read_excel('table1.xlsx')
df2 = pd.read_excel(shared_path + '\\金融市场数据.xlsx')  # 或 pd.read_excel('table2.xlsx')

# 统一日期格式
df1.iloc[:, 0] = pd.to_datetime(df1.iloc[:, 0], format='%Y/%m/%d', errors='coerce')
df2.iloc[:, 0] = pd.to_datetime(df2.iloc[:, 0], errors='coerce')

# 重命名第一列为“日期”方便处理
df1.columns.values[0] = '日期'
df2.columns.values[0] = '日期'

df1['日期'] = pd.to_datetime(df1['日期'])
df2['日期'] = pd.to_datetime(df2['日期'])

print("df1 日期列类型：", df1['日期'].dtype)
print("df2 日期列类型：", df2['日期'].dtype)

# 读取df2的第二个sheet
df1_sheet2 = pd.read_excel(shared_path + '\\夜光卫星遥感数据.xlsx', sheet_name=1)  # sheet_name=1表示第二个sheet

# 统一日期格式并重命名列
df1_sheet2.iloc[:, 0] = pd.to_datetime(df1_sheet2.iloc[:, 0], errors='coerce')
df1_sheet2.columns.values[0] = '日期'
df1_sheet2['日期'] = pd.to_datetime(df1_sheet2['日期'])
print("df1_sheet2 日期列类型：", df1_sheet2['日期'].dtype)

# 合并两个DataFrame，按“日期”对齐
merged_df = pd.merge(df2, df1, on='日期', how='left')
merged_df = pd.merge(merged_df, df1_sheet2, on='日期', how='left')

# 保存结果
merged_df.to_csv('origin_merged_result.csv', index=False, encoding='gbk')
