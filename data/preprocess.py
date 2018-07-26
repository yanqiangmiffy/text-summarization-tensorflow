import json
import jieba
import pandas as pd
stop_words=open('stop_words.txt','r',encoding='utf-8').read().split('\n')
print(stop_words)

def process(filename,target_filename):
    """
    对于文本进行分词，去除停用词
    :param filename:
    :param target_filename:
    :return:
    """
    print(filename+" on process....:\n")

    new_file=open(target_filename,'w',encoding='utf-8')
    article_len=[]
    title_len=[]
    num=0

    with open(filename,'r',encoding='utf-8') as raw_file:
        for line in raw_file:
            num+=1
            data=json.loads(line.strip('\r\n'))

            article_words = []
            for word in jieba.cut(data['article']):
                if word not in stop_words and word!=' ':
                    article_words.append(word)
            article_len.append(len(article_words))
            data['article']=' '.join(article_words)

            title_words=[]
            for word in jieba.cut(data['summarization']):
                if word not in stop_words and word!=' ':
                    title_words.append(word)
            title_len.append(len(title_words))
            data['summarization']=' '.join(title_words)

            # new_data={"summarization":data["summarization"],"article":data["article"]}
            json_str=json.dumps(data, ensure_ascii=False)

            new_file.write(json_str+'\n')
            if num==10000:
                break
    article_len_df=pd.DataFrame(article_len)
    print(article_len_df.describe())

    title_len_df = pd.DataFrame(title_len)
    print(title_len_df.describe())
if __name__ == '__main__':

    process('raw/evaluation_with_ground_truth.txt','preprocessed/valid_data.txt')
    process('raw/train_with_summ.txt','preprocessed/train_data.txt')
