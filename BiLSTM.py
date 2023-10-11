#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install openpyxl


# In[2]:


pip install nltk


# In[3]:


# 查看数据
import pandas as pd
dataset = pd.read_excel('中小学文言文语料.xlsx')
print(dataset)


# In[26]:


# 数据处理
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import torch.utils.data as Data

BATCH_SIZE = 15
LR = 1e-4
NUM_HIDDENS = 256
SEQ_LEN = 40
WORD_DIM = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './'
chinese_data_path = os.path.join(path , '语料.txt')
origin_data , predict_word = [],[]

# vocab
vocab = []
# str to index
s2i = dict()
# index to str
i2s = dict()
index = 0

def get_vocab(path):
    chinese_data_path = path
    with open(chinese_data_path , 'r') as f:
        # print('loading txt...')
        for line in f:
            if len(line) >= 41:
                # 对？进行特殊处理
                line = line.replace("?", "。")
                line = line.replace("？", "。")
                predict_word.append(line[20])
                d = line[:20]+'?'+line[21:40]
                origin_data.append(d)
                for word in line:
                    if word not in vocab:
                        vocab.append(word)
    f.close()
get_vocab(chinese_data_path)

# vocab size
vocab.append('?')
vocab_size = len(vocab)

random.shuffle(vocab)
s2i = {word: i for i, word in enumerate(vocab)}
i2s = {i: word for i, word in enumerate(vocab)}


# In[27]:


# 划分数据集
class Dataset(Data.Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x,y = self.data[index]
        # print(x,y)
        x = [s2i[i] for i in x]
        y = s2i[y]
        return torch.LongTensor(x), torch.tensor(y)

data = list(zip(origin_data,predict_word))
random.shuffle(data)
train_dataset = Dataset(data[:100])
test_dataset = Dataset(data[100:])
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    num_workers=2,
    batch_size=BATCH_SIZE,
    drop_last=True
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    num_workers=2,
    batch_size=BATCH_SIZE,
    drop_last=True
)

len(train_dataset), len(test_dataset)


# In[28]:


# 定义函数
import nltk
EPOCH = 10
import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=WORD_DIM,
            hidden_size=NUM_HIDDENS,
            bidirectional=True
          )
        self.out = nn.Linear(NUM_HIDDENS*2, vocab_size)
    def forward(self,x):
        # print(x.shape)
        # x.shape [batch,seq_len,word_dim]
        # state.shape [1,seq_len,word_dim]
        h_s = torch.randn((1*2,BATCH_SIZE, NUM_HIDDENS),device=device)
        c_s = torch.randn((1*2,BATCH_SIZE, NUM_HIDDENS),device=device)
        output,(_,_)= self.lstm(x.transpose(1,0),(h_s,c_s)) # output.shape [seq_len,batch,num_hiddens]
        chinese = self.out(output[20])  # chinese.shape [batch,vocab_size]

        return chinese

def encode(data,embed):
    return embed(data)

def train_porcess(pred,acc,device):
    # pred.shape [batch,vocab_size]
    # acc.shape [batch]
    # print(pred.shape)
    pred = F.softmax(pred, dim=-1)
    # print(pred.shape)
    pred = pred.argmax(dim=-1).cpu().numpy()
    acc = acc.cpu().numpy()
    hypothesis = [i2s[i] for i in pred.tolist()]
    reference = [i2s[i] for i in acc.tolist()]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    print('预测下一个字:',[i2s[i] for i in pred.tolist()])
    print('实际下一个字:',[i2s[i] for i in acc.tolist()])
    print('BLEU值为：',BLEUscore)


# In[29]:


# 模型实例化
bilstm = BiLSTM(vocab_size)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(bilstm.parameters(), lr=LR)
embed = nn.Embedding(vocab_size, WORD_DIM, device = device)
# 保存整个模型 torch.save(bilstm, "BiLSTM.pth")
torch.save(bilstm.state_dict(),"BiLSTM.pth")
output_model = './BiLSTM.pth'


# In[30]:


# 训练模型
BATCH_SIZE = 15
if __name__ == '__main__':

    bestscore = 0
    # output_model = './BiLSTM.pth'
    state_dict = torch.load(output_model, map_location='cpu')
    # print(checkpoint)
    bilstm.load_state_dict(torch.load("BiLSTM.pth"))
    bilstm.to(device)
    bilstm.train()
    for epoch in range(10):
        print(f'epoch = {epoch}')
        accuracy = 0
        time = 0
        for step, datas in enumerate(train_loader):
            data, label = tuple(t.to(device)for t in datas)
            data = encode(data, embed)
            chinese = bilstm(data)
            loss = criterion(chinese, label.long())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step%10 == 0:
                print(f'loss = {loss}')

            if step%50 == 0:
                train_porcess(chinese, label, device)
        bilstm.eval()
        for step, datas in enumerate(test_loader):
            # print(datas)
            data, label = tuple(t.to(device)for t in datas)
            data = encode(data, embed)
            chinese = bilstm(data)

            pred = F.softmax(chinese, dim=-1)
            pred = pred.argmax(dim=-1).cpu().numpy()
            acc = label.cpu().numpy()
            accuracy += sum(acc==pred)
            time += len(acc)
            accuracy/=time+0.0
        # 百分数形式 print('准确度为：{:.2f}%\n'.format(accuracy * 100))
        print('准确度为：',accuracy)
        print('\n')
        if accuracy>bestscore:
            torch.save(
            {
                'model_state_dict': bilstm.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            }, output_model
          )
            bestscore = accuracy
        bilstm.train()

