import re
import numpy as np
import jieba
import gensim

dict_dim_dict = {
    'sgns.financial.word': 300,
}
stopwords_path = r"C:\Users\lzx\Desktop\大四暑\易方达杯\stopwords\hit_stopwords.txt"
model_path = rf"C:\Users\lzx\Desktop\大四\大四下\毕设\sgns.financial.word\sgns.financial.word"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    return _model


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
    model = get_model()
    token_list = clean_and_tokenize(text)
    token_list = keep_token_in_dict(token_list, model)
    token_list = [model[word] for word in token_list]
    token_list = pad_or_truncate(token_list)
    return token_list
