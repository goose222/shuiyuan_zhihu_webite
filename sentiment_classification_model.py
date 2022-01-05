import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data,vocab
import jieba
import re
import numpy as np
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

df = pd.read_csv('weibo_senti_100k.csv')
print(df.head())

X = []
Y = []

for i,r in df.iterrows():
    Y.append(r['label'])
    X.append(r['review'])

TEXT = data.Field(lower=True, include_lengths=True)
LABEL = data.LabelField()


class PostDataset(data.Dataset):
    def __init__(self, post_field, label_field, posts, labels):
        fields = [('post', post_field), ('label', label_field)]
        examples = []
        self.post_list = posts
        self.label_list = labels

        for i in range(len(self.post_list)):
            examples.append(data.Example.fromlist([self.post_list[i], self.label_list[i]], fields))

        super().__init__(examples, fields)

    @staticmethod
    def sort_key(input):
        return len(input.post)

for x in X:
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    x = re.sub(pattern,' ',x)
    x = jieba.cut(x, cut_all=False)
    x = " ".join(x)
    x = x.split(' ')
    while "" in x:
        x.remove("")
    # print(x)
train = PostDataset(TEXT, LABEL, X, Y)
vector = vocab.Vectors(name="sgns.weibo.bigram")

TEXT.build_vocab(train,vectors=vector)
LABEL.build_vocab(train)
print(TEXT.vocab.vectors)

train_iter = data.BucketIterator(train, batch_size=128, sort_within_batch=True,
                                 shuffle=True)

BATCH_SIZE = 128
hidden_dim = 256
embedding_dim = 300
vocab_size = len(TEXT.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # padding

class bigru_attention(nn.Module):
    def __init__(self):
        super(bigru_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向GRU，//操作为了与后面的Attention操作维度匹配，hidden_dim要取偶数！
        self.bigru = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=self.gru_layers, bidirectional=True)
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # 二分类
        self.fc = nn.Linear(hidden_dim, 2)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, sentence):
        embeds = self.embedding(sentence)  # [seq_len, bs, emb_dim]
        gru_out, _ = self.bigru(embeds)  # [seq_len, bs, hid_dim]
        x = gru_out.permute(1, 0, 2)
        # # # Attention过程，与上图中三个公式对应
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        # # # Attention过程结束

        feat = torch.sum(scored_x, dim=1)
        y = self.fc(feat)
        return y


model = bigru_attention()
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def binary_accuracy(preds, y):
 #   print(preds)
    rounded_preds = preds
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def cal_F1(preds, y):
    rounded_preds = preds
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1:
            if rounded_preds[i] == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if rounded_preds[i] == 1:
                fp = fp + 1
            else:
                tn = tn + 1

    return tp, tn, fp, fn

def train(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    epoch_tp = 0
    epoch_tn = 0
    epoch_fp = 0
    epoch_fn = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.post
        print('training....')
        optimizer.zero_grad()
        print(text)
        print(text.shape)
        predictions = model(text)
        # print(predictions.shape)
        softmax_pred = torch.softmax(predictions,dim=1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(torch.max(softmax_pred, dim=1)[1], batch.label)
        tp, tn, fp, fn = cal_F1(torch.max(softmax_pred, dim=1)[1], batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_tp += tp
        epoch_tn += tn
        epoch_fn += fn
        epoch_fp += fp

    p = epoch_tp / (epoch_tp + epoch_fp)
    r = epoch_tp / (epoch_tp + epoch_fn)
    F1 = 2 * p * r / (p + r)
    print('train: '+'F1:' + str(F1) + " p:" + str(p) + " r:" + str(r))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_model():
    t = time.time()
    loss = []
    acc = []
    val_acc = []
    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_iter)
        # test_acc = evaluate(model,test_iter)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        # print(f'\t Val. Acc: {test_acc * 100:.2f}%')

        loss.append(train_loss)
        acc.append(train_acc)
        torch.save(model, 'sentiment_model.pkl')

    print(f'time:{time.time() - t:.3f}')


sentiment_model = torch.load('sentiment_model.pkl')
def predict_sentiment(texts):
    test_x=[]
    text_y=[0]*len(texts)
    for text in texts:
        x = jieba.cut(text, cut_all=False)
        x = " ".join(x)
        x = x.split(' ')
        while "" in x:
            x.remove("")
        print(x)
        print(type(x))
        test_x.append(x)
    test_data = PostDataset(TEXT,LABEL,test_x,text_y)
    test_iter = data.BucketIterator(test_data, batch_size=128, sort_within_batch=True,
                                     shuffle=True)
    with torch.no_grad():
        for batch in test_iter:
            txt, txt_lengths = batch.post
            predictions = model(txt)
            # print(predictions.shape)
            softmax_pred = torch.softmax(predictions, dim=1)
            res = torch.max(softmax_pred, dim=1)[1]
            print(res)


predict_sentiment(['我觉得很开心哦','我感到太难受了','今天天气好差，真是郁闷'])

