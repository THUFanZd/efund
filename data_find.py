import pandas as pd
import re
import torch
import warnings
warnings.filterwarnings('ignore')
from conf_text_tokenize import passage_tokenize

def extract_year_quarter(text):
    pattern = r'(\d{4})年(第[一二三四]季度)'
    match = re.search(pattern, text)
    if match:
        year = match.group(1)
        quarter = match.group(2)
        return year, quarter
    return None, None

def find_macro_data(df: pd.DataFrame, date_str: str, num_month: int):
    #df = pd.read_csv('../macro_hist.csv', encoding='gbk')
    idx = df[df['日期'] == date_str].index
    if idx[0] + num_month + 1 > len(df):
        return torch.zeros((num_month, len(df.columns)-1))  # -1 因为要去掉日期列
    return torch.FloatTensor(df[idx[0]+1: idx[0]+num_month+1].drop(columns=['日期']).values.astype(float))

def find_conf_data(df: pd.DataFrame, date_str: str, num_month: int):
    #df = pd.read_csv('../pbc_conference.csv')
    df['年份'], df['季度'] = zip(*df['title'].apply(extract_year_quarter))
    date = pd.to_datetime(date_str)
    prev_months = [date - pd.DateOffset(months=i) for i in range(1, num_month+1)]
    
    res = []
    for month in prev_months:
        quarter = (month.month - 1) // 3 + 1
        prev_quarter = quarter - 1 if quarter > 1 else 4
        prev_year = month.year if quarter > 1 else month.year - 1
        
        mask = (df['年份'] == str(prev_year)) & (df['季度'] == f'第{["一","二","三","四"][prev_quarter-1]}季度')
        matched = df[mask]
        
        if not matched.empty:
            res.append(matched.iloc[0]['content'])
        else:
            # 添加空字符串作为占位符
            res.append("")

    res = [torch.FloatTensor(passage_tokenize(r)) for r in res]
    return torch.stack(res)

def find_merged_data(df: pd.DataFrame, date_str: str, num_month: int):
    df['日期'] = pd.to_datetime(df['日期'])
    date = pd.to_datetime(date_str)
    prev_months = [date - pd.DateOffset(months=i) for i in range(1, num_month+1)]
    
    res = []
    for month in prev_months:
        month_data = df[(df['日期'].dt.year == month.year) & (df['日期'].dt.month == month.month)]
        month_data = df[month_data.index[0]: month_data.index[0] + 30]
        # 去掉第一列
        month_data = month_data.drop(columns=['日期'])
        # 强制转化dtype不是数字的为数字
        month_data = month_data.apply(pd.to_numeric, errors='coerce')
        # print(month_data.dtypes)
        numeric_data = month_data.to_numpy()
        res.append(torch.FloatTensor(numeric_data))

    res = torch.stack(res)
    return res


if __name__ == '__main__':
    df = pd.read_csv('../merged_result.csv', encoding='gbk')
    res = find_merged_data(df, "2020/12/31", 6)
    print(res.shape)
    exit()
    df = pd.read_csv('../macro_hist.csv', encoding='gbk')
    res = find_macro_data(df, '2025/7/31', 6)
    print(res.shape)
    exit()
    # df = pd.read_csv('../merged_result.csv', encoding='gbk')
    df = pd.read_csv('../pbc_conference.csv')
    res = find_conf_data(df, '2025/03/28', 6)
    print(type(res))
    print(res.shape)
    exit()
    # merged result
    merged_result_df = pd.read_csv('../merged_result.csv', encoding='gbk')
    # pbc conference
    pbc_conference_df = pd.read_csv('../pbc_conference.csv', encoding='gbk')

    import chardet
    # 检测文件的编码
    with open('../pbc_conference.csv', 'rb') as f:
        result = chardet.detect(f.read())
        print(result['encoding'])

