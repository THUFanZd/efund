import pandas as pd

res_df = pd.read_excel('../待提交的预测结果及评价指标.xlsx')
c = res_df.iloc[:, 0]
print(c)
print(c[0])
