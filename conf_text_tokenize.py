import re
import numpy as np
import jieba
import gensim

dict_dim_dict = {
    'sgns.financial.word': 300,
}

pbc_conference_path = rf"C:\Users\lzx\Desktop\大四暑\易方达杯\人民银行文本\pbc_conference.csv"
stopwords_path = r"C:\Users\lzx\Desktop\大四暑\易方达杯\stopwords\hit_stopwords.txt"
model_path = rf"C:\Users\lzx\Desktop\大四下\毕设\sgns.financial.word\sgns.financial.word"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)


def clean_text(text):
    text = re.sub(r'[^\w\s!?]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def clean_and_tokenize(text):  # str转化为token的list
    nothing = "   \t\n"  # 包含空格、制表符和换行符
    text = clean_text(text)
    seg_list = jieba.cut(text)
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])

    token_list = [x for x in seg_list if x not in nothing and x not in stopwords]
    return token_list
   

def keep_token_in_dict(token_list, model):
    vectors = [word for word in token_list if word in model]
    return vectors


# 填充或截断
def pad_or_truncate(sentence, max_length=512, vector_dim=300):
    if len(sentence) > max_length:
        return sentence[:max_length]  # 截断
    else:
        return sentence + [np.zeros(vector_dim)] * (max_length - len(sentence))  # 填充


def passage_tokenize(text, args=None):
    token_list = clean_and_tokenize(text)
    token_list = keep_token_in_dict(token_list, model)
    token_list = [model[word] for word in token_list]
    token_list = pad_or_truncate(token_list)
    return token_list

if __name__ == '__main__':
    passage = '中国人民银行2025年3月28日发布的《关于2025年3月28日至2026年3月27日中国人民银行发布的关于金融机构的通知》（以下简称《通知》），通知指出，为了支持金融机构更好地服务客户，提高客户满意度，2025年3月28日至2026年3月27日，中国人民银行将在金融机构业务范围内，严格按照《通知》的要求，对金融机构的业务进行调整和优化。'
    token_list = passage_tokenize(passage)
    print(token_list[0])
    print(len(token_list))
    print(len(token_list[0]))

    exit()
