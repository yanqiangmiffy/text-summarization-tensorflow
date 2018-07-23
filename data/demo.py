import json
new_file=open('train_data.txt','w',encoding='utf-8')
with open('train_with_summ.txt','r',encoding='utf-8') as raw_file:
    for line in raw_file:
        data=json.loads(line.strip('\r\n'))
        # new_data={"summarization":data["summarization"],"article":data["article"]}
        json_str=json.dumps(data, ensure_ascii=False)
        new_file.write(json_str+'\n')
