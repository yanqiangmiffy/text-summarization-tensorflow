import json
import jieba
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

train_path='data/train_data.txt'
valid_path='data/evaluation_with_ground_truth.txt'

def clean_str(sentence):
    """
    去除数字和字母
    :param sentence:
    :return:
    """
    sentence = re.sub(r"[A-Za-z(),<>!?\'\`]+", " ", sentence)  # 去除英文字母和英文标点符号
    sentence = re.sub(r"[0-9]+", "num", sentence)  # 去除数字
    # sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", sentence)
    return sentence

def get_text_list(data_path,toy):
    with open(data_path,'r',encoding='utf-8') as f:
        if not toy:
            return list(map(lambda x:clean_str(x['article'].strip()),f.readlines()))
        else:
            return list(map(lambda x:clean_str(x['article'].strip()),f.readlines()))[:45000]


if __name__ == '__main__':
    data_path='data/train_data.txt'
    # with open(data_path, 'r', encoding='utf-8') as raw_file:
    #     for line in raw_file:
    #         line = line.strip('\r\n')
    #         data=json.loads(line.strip())
    #         sen=clean_str(data['article'])
    #         print(sen)

    text_list=get_text_list(data_path,toy=False)
    print(text_list)