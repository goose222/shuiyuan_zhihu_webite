import tensorflow as tf
import os
import json
from bert_serving.client import BertClient

filename='weibo_senti_100k.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 读入tsv文件，生成所有句子的vector
# 命令行输入这句话先：
#bert-serving-start -model_dir /project/bert/chinese-L-12_H768_A-12
# bert-serving-start -model_dir D:\PycharmProjects\softwareprogram\chinese_L-12_H-768_A-12 -num_worker=1 -max_seq_len=64
def sen2vec(filename):
    f = open(filename, encoding='utf-8')
    data = f.readlines()
    veclist = []
    i = -1
    for line in data:
        i = i+1
        if i == 0:
            continue
        txt = ''
        for j in range(len(line)):
            if line[j]=="," :
                txt = line[j+1:len(line)]
                break

        # 生成句向量
        bc = BertClient()
        print(txt)
        vector = bc.encode([txt]).tolist()
        veclist.append(vector)

        # with open('vector.json", "w") as f:
        #     json.dump(vector, f)
    return veclist

print(sen2vec(filename))