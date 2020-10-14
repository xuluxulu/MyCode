#----处理语言的模型：RNN,LSTM，GRU-----
#为了防止出现梯度爆炸，使用Gradient Clippint (梯度截断)
#保存和加载训练好的模型
import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random
import torch.nn as nn


USE_CUDA = torch.cuda.is_available()   #判断gpu是否存在

#将random。seed()固定在一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)


BATCH_SIZE = 32   #一个batch_size中的句子
EMBEDDING_SIZE = 100    #把输入数据映射的维度
MAX_VOCAB_SIZE = 50000      #句子的数量
HIDDEN_SIZE = 100
NUM_EPOCHS = 2

TEXT = torchtext.data.Field(lower = True)    #输入的单词(文本)都是小写:即对输入文本的要求

VOCAB_SIZE = len(TEXT.vocab)
#获得模型的数据集
train,val,test = torchtext.dataset.LanguageModelingDataset.split(path = '.',train = "text8.train.txt",validation = "text8.val.txt",
                                                test = "text8.test.txt",text_field = TEXT)
TEXT.build_vocab(train,max_size = MAX_VACAB_SIZE)    #获得出现频率最高的前MAX_VACAB_SIZE单词
device = torch.device("CUDA" if USE_CUDA else "cpu")   #设定模型运行的设备

#bptt_len指定反向传播的长度，repeat指对数据只训练一遍就不再重复操作，shffle打乱数据
#----把数据分成若干个batch_size
train_iter,val_iter,test_iter = torchtext.data.BBTTIterator.splits(
    (train,val,test),batch_size = BATCH_SIZE,device = device,bptt_len = 50, repeat = False,shuffle = True )



class RNNModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(RNNModel,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)    #一般一行表示一个单词
        self.lstm = nn.LSTM(embed_size,hidden_size)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.hidden_size = hidden_size
    def forward(self,text,hidden):
        #text : seq_length*batch_size
        emb = self.embed(text)   #seq_length*batch_size*embed_size

        output,hidden = self.lstm(emb,hidden)
        #output:seq_length*batch_size*hidden_size
        #hidden:(1*batch_size*hidden_size,1*batch_size*hidden_size)

        #将output数据的维度转换成二维的
        output = output.view(-1,output.shape[2])  #(seq_length*batch_size,hidden_size)

        out_vocab = self.linear(output)   #对结果进行解码  #(seq_length*batch_size,hidden_size)
        return out_vocab.view(output.size(0),output.size(1),out_vocab.size(-1)),hidden

    def init_hidden(self,bsz,requires_grad = True):
        weight = next(self.parameters())
        return [(weight.new_zeros((1,bsz,self.hidden_size),requires_grad = True)),(weight.new_zeros((1,bsz,self.hidden_size),requires_grad = True))]


model = RNNModel(vocal_size = len(TEXT.vocab),embed_size = EMBEDDING_SIZE,hidden_size = HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)   #将模型放到gpu上



def repackage_hidden(h):
    if isinstance(h,torch.Tensor):   #若h是一个Tensor（即是一个元素）
        return h.detach()     #让h与前面没有关系，成为一个全新的节点
    else:
        return tuple(repachage_hidden(v) for v in h)   #若h中有若干个节点，则把每个节点都作为一个新的开始

loss_fn = nn.CrossEntropyLoss()   #使用交叉熵损失函数
learning_rate =0.001
GRAD_CLIP = 5.
val_loss = []
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

#用来降学习率的方法
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.5)   #每次降低0.5被

def evaluate(model,data):
    model.eval()
    total_loss =0
    tatal_count = 0
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE,requires_grad = False)
        for i,batch in enumerate(it):
            data,target = batch.text,batch.target
            hidden = repackage_hidden(hidden)
            output,hidden = model(data,hidden)
            loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))  #此处的loss被平均过
            total_loss +=loss.item()*np.multiply(*data.size())  #让均值乘于data的数量
            total_count += np.multiply(*data.size())
    loss = total_loss/total_count
    model.train()   #最后需要把模型改成训练的方式，因为此过程还是在训练的过程
    return loss


for epoch in range(NUM_EPOCHS):
    model.train()   #模型训练
    it = iter(train_iter)   #获得所有训练数据的batch
    hidden = model.init_hidden(BATCH_SIZE)   #初始化hidden
    for i,batch in enumerate(it):   #获得每一个批次的数据
        data,target = batch.text,batch.target
        hidden = repackage_hidden(hidden)  #获得一个全新的hidden
        output,hidden = model(data,hidden)      #在这个过程中，hidden的参数在不停的改变

        loss = loss_fn(output.view(-1,VOCAB_SZIE),target(-1))    #batch_size*target_class_dim,batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)  #对权重进行截断
        optimizer.step()

        if i%100==0:
            print(i,loss.item())

        ##对模型进行保存
        if i%100:   #运行1000次后对结果，使用验证数据来验证数据
            val_loss = evaluate(model,val_iter)
            if len(val_losses)==0 or val_loss<min(val_losses):
                torch.save(model.state_dict(),'1m.pth')
            else:
                scheduler.step()    #更改一次学习率
            val_losses.append(val_loss)


#加载训练好的模型
best_model = RNNModel(vocab_size = len(TEXT.vocab),embed_size = EMBEDDING_SIZE,hidden_size = HIDDEN_SIZE)   #初始化模型
if USE_CUDA:
    model = best_model.to(device)   #把模型放到gpu上
best_model.load_state_dict(torch.load('1m.pth'))  #加载模型
#使用训练好的模型进行测试数据
val_loss = evaluate(best_model,test_iter)
print(val_loss)

###-----利用训练好的模型产生一些句子
hidden = best_model.init_hidden(1)  #随机产生一个hidden_tate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE,(1,1),dtype = torch.long).to(device)   #产生数据，并放到gpu上
words = []
for i in range(100):
    #output是输出结果,维度是（1，1，维度）
    output,hidden = best_model(input,hidden)   #随着循环迭代，上一个结果的输出是此次结果的输入

    #logits exp
    word_weights = output.squeeze().exp().cpu()   #把结果放到gpu上,squeeze()是把所有维度为1的都扔掉，如有一个[10,1,1]的tensor,经过转换变成[10]的tensor
    word_idx = torch.multinomial(word_weights,1)[0]   #得到一个单词的index，作用和argmax()一样的，得到最大值的索引

    input.fill_(word_idx)    #把当前单词的索引放到input，然后用input来预测下一个结果

    word = TEXT.vocab.itos[word_idx]   #通过索引获得一个单词
    words.append(word)
print(" ".join(words))








