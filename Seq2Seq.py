#---构建seq2seq的网路结构：没有Attention机制--------
#----此程序中会有Bug的

import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk  #进行分词的库

def load_data(in_file):
    cn = []   #中文
    en = []   #英文
    num_examples = 0
    with open(in_file,'r') as f:  #打开文件
        for line in f:  #处理每一行
            line = line.strip('\t')
            en.append(['BOS'] + nltk.word_tokenize(line[0].lower()) + ['EOS'])   #把英文都转成小写，nltk.word_tokenize()把每一句分成若干个单词
            cn.append(['BOS'] + [c for c in line[1] ] + ['EOS'])
    return en,cn

#--1.加载数据
train_file = "nmt/en-cn/train.txt"
dev_file = "nmt/en-cn/dev.txt"   #验证数据集
train_en ,train_cn = load_data(train_file)
dev_en,dev_cn = load_data(dev_file)

#--2.构造单词表
UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences,max_words = 50000):
    word_count = Counter()  #声明一个计数器，计算每个单词出现的次数
    for sentence in sentences:
        for s in sentence:
            word_count[s] +=1
    ls = word_count.most_common(max_words)   #获得出现频率最高的前max_words的单词：返回的是字典格式
    total_words = len(ls) +2  #因为又在字典中加入UNK_IDX 和PAD_IDX
    #通过名字可以找到对应的索引
    word_dict = {w[0]:index+2 for index,w in enumerate(ls)}  #将整体的索引向后偏移两个单位
    word_dict['UNK'] = UNK_IDX
    word_dict['PAD'] = PAD_IDX
    return word_dict,total_words


#通过名字找索引
en_dict,en_total = build_dict(train_en)
cn_dict,cn_total = build_dict(train_cn)


###此处.items()获得多个值，功能类似于keys()得到一个字典的所有key
inv_en_dict = {v:k for k,v in en_dict.items()}  #字典格式：通过索引得到对应的字符串
inv_cn_dict = {v:k for k,v in cn_dict.items()}



#--3.把所有的单词都转换成数字:即把每一个单词转换成对应的索引
def encode(en_sentences,cn_sentences,en_dict,cn_dict,sort_by_len=True):
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w,0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w,0) for w in sent] for sent in cn_sentences]

    def len_argsort(seq):
        return sorted(range(len(seq)),key = lambda x:len(seq(x)))
    #把句子按英语句子的长度进行排序;把中文和英文按同样的顺序进行排序
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
    return out_en_sentences,out_cn_sentences

train_en,train_cn = encode(train_en,train_cn,en_dict,cn_dict)
dev_en,dev_cn = encode(dev_en,dev_cn,en_dict,cn_dict)

#--4.把句子分成batch
def git_minibatches(n,minibatch_size,shuffle = False):
    idx_list = np.arange(0,n,minibatch_size)    #获得每一个batch_size开始的索引
    if shuffle:
        np.random.shuffle(ind_list)   #打乱数组中各个数字的顺序
    minibatches = []   #保存的是每个batch_size中的索引
    if idx in idx_list:
        minibatches.append(np.arange(idx,min(idx+minibatches_size,n)))
    return minibatches

#准备数据:把每个批次的数据放到数组中
def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]    #得到所有句子的长度
    n_samples = len(seqs)  #句子的数量
    max_len = np.max(lengths)   #获得最长句子的长度

    x = np.zeros((n_samples,max_len)).astype('int32')
    x_lengths = np.array(lengths).astype('int32')   #将list转换成数组
    for idx,seq in enumerate(seqs):
        x[idx,:lengths[idx]] = seq
    return x,x_lengths

def gen_examples(en_sentences,cn_sentences,batch_size):
    minibatches = get_minibatches(len(en_sentence),batch_size)
    all_ex = []

    for minibatch in minibatches:
        #将句子分批次保存起来
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        #建立英文和中文之间的对应关系
        mb_x,mb_x_len = prepared_data(mb_en_sentences)
        mb_y,mb_y_len = prepared_data(mb_cn_sentences)
        all_ex.append((mb_x,mb_x_len,mb_y,mb_y_len))
    return all_ex

batch_size = 64
train_data = gen_examples(train_en,train_cn,batch_size)
random.shuffle(train_data)
dev_data = gen_examples(dev_en,dev_cn,batch_size)   #对验证数据进行划分


#定义一个没有attention的seq2seq
class PlainEncoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout = 0.2):
        super(PlainEncoder,self).__init__()
        self.embed = nn.embedding(vocab_size,hidden_size)
        self.rnn - nn.GRU(hidden_size,hidden_size,batch_first = True)
        self.dropout = nn.Dropout(dropout)
    def forward(self,y,y_lengths,hid):
        #对batch中的句子按长度排序
        sorted_len,sorted_idx = y_lengths.sort(0,descending = True)   #按降序排列:排序后的结果，排序后的结果中每个句子在未排序时的索引
        y_sorted = y[sorted_idx]  #对输入句子进行重新排序
        hid = hid[sorted_idx]
        embedded = self.dropout(self.embed(y_sorted))
        embedded = torch.cat([embedded,hid.unsqueeze(0).expand_as(embedded)],2)

        #以下程序就是对变长的程序进行高效的处理
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().cpu().data.numpy(),batch_first = True)   #
        out,newhid = self.rnn(packed_embedded,hid)
        out,_ = nn.utils.rnn.pad_packed_sequence(out,batch_first = True)

        _,original_idx = sorted_idx.sort(0,descending = False)
        out = out[original_idx.long()].contiguous()
        newhid = newhid[original_idx.long()].contiguous()

        out = torch.cat([out,hid.unsqueeze(0).expand_as(out)],2)
        out = F.log_softmax(self.fc(out))
        return out,newhid


class PlainDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout = 0.2):
        super(PlainDecoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(2*hidden_size,hidden_size,batch_first = True)
        self.out = nn.Linear(2*hidden_size,vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,y,y_lengths,hld):
        sorted_len,sorted_idx = y_lengths.sort(0,descending = True)
        x_sorted = x[sorted_idx]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().cpu().data.numpy(),batch_first = True)
        packed_out,hid = self.rnn(packed_embedded)
        out,_ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first = True)

        _,original_idx = sorted_idx.sort(0,descending = False)
        out = out[original_idx.long()].contiguous()
        hid = hid[original_idx.long()].contiguous()

        return out,hid[[-1]]

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion,self).__init__()
    def forward(self,input,target,mask):
        input = input.contiguous().view(-1,input.size(2))
        target = target.contiguous().view(-1,1)
        mask = mask.contiguous().view(-1,1)
        output = -input.gather(1,target) * mask
        output = torch.sum(output)/torch.sum(mask)
        return output

#定义Seq2Seq的网络模型结构
class PlainSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(PlainSeq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,x,x_length,y,y_length):
        encoder_out,hid = self.encoder(x,x_length)
        output,hid = self.decoder(y,y_length,hid)
        return output,None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100
encoder = PlainEncoder(vocab_size = en_total_words,
                       hidden_size = hidden_size,
                       dropout = dropout)
decoder = PlainDecoder(vocab_size = cn_total_words,
                       hidden_size = hidden_size,
                       dropout = dropout)
model = PlainSeq2Seq(encoder,decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())

#训练程序
def train(model,data,num_epochs =30):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.

        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0]=1

            mb_pred,attn = model(mb_x,mb_x_len,mb_input,mb_y_len)
            mb_out_mask = torch.arange(mb_y_len.max().item(),device = device)[None,:] < mb_y_len[:,None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred,mb_output,mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item()*num_words
            total_num_words += num_words
            #更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),5.)  #防止梯度爆炸，使用梯度裁剪的方式
            optimizer.step()
            if epoch%5==0:
                evaluate(model,dev_data)   #每5次，使用验证集测试一次模型

def evaluate(model,data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0]=1

            mb_pred,attn = model(mb_x,mb_x_len,mb_input,mb_y_len)
            mb_out_mask = torch.arange(mb_y_len.max().item(),device = device)[None,:] < mb_y_len[:,None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred,mb_output,mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item()*num_words
            total_num_words += num_words
            print('平均损失为:',tatal_loss/total_num_words)


train(model,train_data,num_epochs = 200)



