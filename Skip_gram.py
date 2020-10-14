#--------Skip_gram的词向量表示：即用一个词的周围词来表示此单词，设定一个模型实现此功能
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud    #对数据集操作

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


USE_CUDA = torch.cuda.is_available()    #判断cuda(GPU)是否存在
#seed()作用：指定随机数的产生时所使用算法开始的整数，若设定相同的值，则每次初始化的值都是一样的
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

#设定一些hyper parameters(超参数)
C = 3   #context window (表示中心词周围空间的大小)
K = 100 #number of negative samples(负样本的数量)
NUM_EPOCHS = 1000   #中心词的大小
MAX_VOCAB_SIZE = 30000   #词向量的大小
BATCH_SIZE = 128   #批量大小
LEARNING_RATE = 0.2 #学习率的大小
EMBEDDING_SIZE = 100   #把输入数据映射成的维度




with open('text8.train.txt','r') as fin:
    text = fin.read()   #读取文本的所有,一个字符串

text = text.split()   #对字符串以’ ‘的形式进行拆分，产生list

#vocab是一个字典，key表示单词，value表示每个单词出现的次数
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))    #统计出现次数最高的前MAX_VOCAB_SIZE-1的单词的次数，并以字典的形式展示，key是单词，value是出现的次数
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))     #表示不常见的单词出现的次数

idx_to_word = [word for word in vocab.keys()]   #将vocab中所有单词的key放到list
word_to_idx = {word:i for i,word in enumerate(ind_to_word)}    #通过单词找到索引

#得到每个单词出现的频率
word_counts =np.array([count for count in vocab.values()],dtype = np.float32)
word_fre = word_counts/np.sum(word_counts)
#根据原论文再进行下述操作
word_fre = word_fre**(3./4.)
word_fre = word_counts/np.sum(word_counts)
VOCAB_SIZE = len(idx_to_word)   #在一个重新获得原文中单词的数量




#定义一个对数据集操作的类：操作数据集
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,text,word_to_idx,idx_to_word,word_fre,word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_ind.get(word,word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_fre = torch.Tensor(word_fre)   #将list转换成tensor
        self.word_counts = torch.Tensor(word_counts)
    def __len__(self):
        return len(self.text_encoded)   #获得text中单词的数量

    def __getitem__(self,idx):
        center_word = self.text_encoded(idx)   #根据索引获得中心词
        #获得中心词周围词汇的索引
        pos_indices = list(rang(idx-C,idx)) + list(range(idx+1,idx+C+1))    #在range(a,b)中b值是不取的
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]   #正样本:中心单词的周围单词
        neg_words = torch.multinomial(self.word_fre,K*pos_words.shape[0],True) #负样本：负例采样单词
        return center_word,pos_words,neg_words


dataset = WordEmbeddingDataset(text,word_to_idx,idx_to_word,word_fre,word_counts)
#使用此接口读取数据集，并将数据集打乱，采用4个子进程读取数据，数据集的batch_size大小为BATCH_SIZE
dataloader = tud.DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = True,num_workers= 4)   #自动调用__getitem__（）函数



#定义模型
class EmbeddingModel(nn.Module):   #embedding的作用是自动学习到输入空间到distribute representation空间的映射F
    def __init__(self,vocab_size,embed_size):
        super(EmbeddignModel,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        intrange = 0.5/self.embed_size

        #对各种变量进行embedding操作
        self.in_embed = nn.Embedding(self.vocab_size,self.embed_size)  #即把vocab_size大小降到embed_size大小
        self.in_embed.weight.data.uniform_(-intrange,intrange)    #对数据进行归一化
        self.out_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.out_embed.weight.data.uniform_(-intrange,intrange)

    def forward(self,input_labels,pos_labels,neg_labels):  #正确的中心词汇，正确的周围词汇，正确的负样例词汇
        #input_label:[batch_size]
        #pos_labels:[batch_size,(window_size*2)]
        #neg_labels:[batch_size,(window_size*2*K)]

        #定义输入输出数据
        input_embedding = self.in_embed(input_labels)   #[batch_size,embed_size]
        pos_embedding = self.in_embed(pos_labels)    #[batch_size,(window_size*2),embed_size]
        neg_embedding = self.in_embed(neg_labels)   #[batch_size,(window_size*2*K),embed_size]
        input_embedding = input_embedding.unsqunze(2)    #[batch_size,embed_size,1]   #多加了一个维度

        #对下列数据做点乘操作：建立各种输出与中心词之间的关系
        pos_dot  = torch.bmm(pos_embedding,input_embedding).squenze(2)   #[batch_size,(window_size*2),1]
        neg_dot = torch.bmm(neg_embedding,-input_embedding).squenze(2)  #[batch_size,(window_size*2*K),1]

        #定义损失函数
        log_pos = F.logsigmoid(pos_dot).sum(1)    #sum(1)：对维度为1的求和
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg
        return -loss    #得到损失函数

    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()   #获得输入数据的权重


#定义模型并放到gpu上
model = EmbedingModel(VOCAB_SIZE,EMBEDDING_SIZE)   #获得模型
if USE_CUDA:
    model = model.cuda()   #将模型放到gpu上


optimizer = torch.optim.SGD(model.parameters(),lr = LRARNING_RATE)  #定义一个优化器
#训练模型
for e in range(NUM_EPOCHS):
    for i ,(input_labels,pos_labels,neg_labels) in enumerate(dataloader):
        #将数据都变成long整型
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        #将所有的张量都放到gpu上
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

            optimizer.zero_grad()   #每一次都将优化器清空
            loss = model(input_labels,pos_labels,neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i%100==0:
                print(i,loss.item())   #.item()获得张量中的值














