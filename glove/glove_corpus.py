from utils import get_text_list
# 路径设置
train_path = 'data/preprocessed/train_data.txt'
valid_path = 'data/preprocessed/valid_data.txt'
glove_corpus_path = 'glove_corpus'

train_article=get_text_list(train_path,False,flag='article')
train_title=get_text_list(train_path,False,flag='title')

valid_article=get_text_list(valid_path,False,flag='article')
valid_title=get_text_list(valid_path,False,flag='title')

text_list=train_article+train_title+valid_article+valid_title

with open(glove_corpus_path,'w',encoding='utf-8') as out_data:
    for text in text_list:
        out_data.write(text+'\n')