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

def get_text_list(data_path,toy,flag):
    """
    返回文章和标题文本列表
    :param data_path: 文件路径
    :param toy: 是否使用全部数据集
    :param flag: 返回文章或者标题
    :return:
    """
    if flag=='article':
        with open(data_path,'r',encoding='utf-8') as f:
            if not toy:
                return list(map(lambda x:clean_str(json.loads(x.strip())['article']),f.readlines()))
            else:
                return list(map(lambda x: clean_str(json.loads(x.strip())['article']), f.readlines()))[:45000]
    if flag=='title':
        with open(data_path,'r',encoding='utf-8') as f:
            if not toy:
                return list(map(lambda x:clean_str(json.loads(x.strip())['summarization']),f.readlines()))
            else:
                return list(map(lambda x: clean_str(json.loads(x.strip())['summarization']), f.readlines()))[:45000]

def build_dict(step,toy=False):
    """
    构建词典
    :param step: 是否为训练
    :param toy:
    :return:
    """
    if step=='train':
        train_article_list=get_text_list(train_path,toy,flag='article')
        train_title_list=get_text_list(train_path,toy,flag='title')

        words=list()
        for sentence in train_article_list+train_title_list:
            for word in sentence.split(' '):
                words.append(word)
        word_counter=collections.Counter(words).most_common()
        word_dict=dict()
        word_dict["<padding>"]=0
        word_dict["<unk>"]=1
        word_dict["<s>"]=2
        word_dict["</s>"]=3
        for word,_ in word_counter:
            word_dict[word]=len(word_dict)

        with open('result/word_dict.pkl','wb') as out_data:
            pickle.dump(word_dict,out_data,pickle.HIGHEST_PROTOCOL)
    elif step=='valid':
        with open('result/word_dict.pkl','rb') as in_data:
            word_dict=pickle.load(in_data)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 1000
    summary_max_len = 20

    return word_dict,reversed_dict,article_max_len,summary_max_len

def build_dataset(step,word_dict,article_max_len,summary_max_len,toy=False):
    """
    创建数据集，将文本转为数字索引表示
    :param step:
    :param word_dict:
    :param article_max_len: 文章最大长度
    :param summary_max_len: 标题最大长度
    :param toy:
    :return:
    """
    if step=='train':
        article_list=get_text_list(train_path,toy,flag='article')
        title_list=get_text_list(train_path,toy,flag='title')
    elif step=='valid':
        article_list=get_text_list(valid_path,toy,flag='article')
        title_list=get_text_list(valid_path,toy,flag='article')
    else:
        raise  NotImplementedError
    x=list(map(lambda d:d.split(' '),article_list))
    x=list(map(lambda d:list(map(lambda w:word_dict.get(w,word_dict["<unk>"]),d)),x))
    x=list(map(lambda d:d[:article_max_len],x))
    x=list(map(lambda d:d+(article_max_len-len(d))*[word_dict["<padding>"]],x))

    y = list(map(lambda d: d.split(' '), title_list))
    y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
    y = list(map(lambda d: d[:(summary_max_len - 1)], y))

    return x,y

def batch_iter(inputs,outputs,batch_size,num_epochs):
    """
    生成批数据
    :param inputs:
    :param outputs:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    inputs=np.array(inputs)
    outputs=np.array(outputs)

    num_batches_per_epoch=(len(inputs)-1)//batch_size+1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,len(inputs))
            yield inputs[start_index:end_index]

def get_init_embedding(reverse_dict,embedding_size):
    glove_file='glove/glove.300d.txt'
    word2vec_file=get_tmpfile("word2vec.format.vec")
    glove2word2vec(glove_file,word2vec_file)
    print("loading Glove vectors...")
    word2vecs=KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list=list()
    for _,word in sorted(reverse_dict.items()):
        try:
            word_vec=word2vecs.word_vec(word)
        except KeyError:
            word_vec=np.zeros([embedding_size],dtype=np.float32)

        word_vec_list.append(word_vec)

    word_vec_list[2]=np.random.normal(0,1,embedding_size)
    word_vec_list[3]=np.random.normal(0,1,embedding_size)

    return np.array(word_vec_list)
if __name__ == '__main__':
    # with open(data_path, 'r', encoding='utf-8') as raw_file:
    #     for line in raw_file:
    #         line = line.strip('\r\n')
    #         data=json.loads(line.strip())
    #         sen=clean_str(data['article'])
    #         print(sen)

    # train_article_list=get_text_list(train_path,toy=False,flag='article')
    train_title_list=get_text_list(train_path,toy=False,flag='title')
    print(train_title_list)