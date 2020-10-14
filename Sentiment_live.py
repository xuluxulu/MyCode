#-----实现情感分析，分析电影的评论是积极还是消极的-------
##Word Avaraging实现原理：对每个单词使用embedding 映射得到word embedding vector，然后通过一句话中的所有vector做个平均，就是整个句子的vector,
#接下来就是对整个句子的vector传入linear（可能包含多层）中做分类即可
import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import random

##1. 对所有的单词进行向量划分，并对整个数据集进行划分不同的测试集
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
BATCH_SIZE = 64
N_EPOCH = 10


TEXT = data.Field(tokenize = 'spacy')   #对输入句子进行分割成单词，或把文章分割成句子
LABEL = data.LabelField(dtype = torch.float)    #文本的标签
train_data,test_data = datasets.IMDB.splits(TEXT,LABEL)    #划分数据集：平分数据集，即二者的数据量是一样的

#每一个train_data或者valid_data包含两个部分：一部分是文本(text)都是向量格式  train_data.text，另一半是label（也是向量格式）   train_data.label
train_data,valid_data = train_data.split(random_state = random.seed(SEED))   #将训练集划分成训练集和验证集：默认7：3

#构建词向量:每一个单词都构建成一个向量
TEXT.build_vocab(trian_data,max_size = 25000,vectors = 'glove.6B.100d',unk_init = torch.Tensor.normal_)   #把模型的单词初始化glove会加快模型收敛速度
LABEL.build_vocab(train_data)


# print(TEXT.vocab.itos[:10])      #获得频率最高的前10个单词，itos指index找string,即index to string
# print(LABEL.vocab.stoi)      #stoi指通过string找index,即 string to index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #设定运行设备

#对不同测试进行批次划分:BucketIterator是把长度差不多的句子的长度划分到一个batch中，确保每一个batch中不会出现太多的padding,并自动打乱句子（此句子是完整的+label）
train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size = BATCH_SIZE,
    device = device
     )
# batch = next(train_iterator)
# print([TEXT.vocab.itos[i] for i in batch.text[:,0]])   #在当前batch中，通过index获得对应的文本

###  2. 使用word Averaging模型做情感类别分类：先对每个单词转换成向量，然后对整个句子所有单词的向量取均值（就是这个句子的向量），之后把这个句子的向量传到网络模型中做分类
class WordAVGModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,output_size,pad_idx):
        super(WordAVGModel,self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size,padding_idx = pad_idx)   #一个单词对应一行
        self.linear = nn.Linear(embedding_size,output_size)
    def forward(self,text):    #默认就会调用这个函数
        #将单词映射成向量
        embeded = self.embed(text)   #[seq_len,batch_size,embedding_size]
        #对数据的维度进行转换
        embeded = embeded.permute(1,0,2)   #将维度为0和1的进行转换
        pooled = F.avg_pool2d(embeded,(embeded.shape[1],1)).squeeze()  #(embeded.shape[1],1)是卷积核，进行池化，其中embeded.shape[1]表示的是列
        return self.linear(pooled)


VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)   #pad_token得到的是pad的对应的索引

model = WordAVGModel(vocab_size = VOCAB_SIZE,
                     embedding_size = EMBEDDING_SIZE,
                     output_size = OUTPUT_SIZE,
                     pad_idx = PAD_IDX)


# #获得模型的参数个数
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)  #p.requires_grad表示此参数是可以更新的，p.numel()获得参数的个数


###3. glove初始化模型可以加快模型的收敛速度
pretrained_embedding = TEXT.vocab.vectors  #得到所有向量
model.embed.weight.data.copy_(pretrained_embedding)
#得到unk的索引
UNK_IDX = TEXT.vocab.stoi(TEXT.unk_token)
#将下列两个权重转换成0:即进行初始化
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)



###4. 定义优化器和损失函数
optimizer  = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()   #损失函数：多标签分类问题
model = model.to(device)
crit.to(device)


###5.计算准确率
def binary_accuracy(preds,y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()  #.float()将布尔型转换成float
    acc = correct.sum()/len(correct)
    return acc

###6.训练模型
def train(model,iterator,crit):
    epoch_loss,epoch_acc = 0.,0.
    total_len = 0
    model.train()
    for batch in iterator:
        preds = model(batch.text).squeeze()   #将维度为1的去掉
        loss = crit(preds,batch.label)
        acc = binary_accuracy(preds,batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss +=loss.item()*len(batch.label)
        epoch_acc +=acc.item()*len(batch.label)
        total_len +=len(batch.label)
    return epoch_loss/total_len,epoch_acc/total_len

###7. 验证模型
def validate(model,iterator,crit):
    epoch_loss,epoch_acc = 0.,0.
    total_len = 0
    model.val()
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds,batch.label)
        acc = binary_accuracy(preds,batch.label)

        epoch_loss +=loss.item()*len(batch.label)
        epoch_acc +=acc.item()*len(batch.label)
        total_len +=len(batch.label)
    model.train()
    return epoch_loss/total_len,epoch_acc/total_len

best_valid_acc = 0
#--开始训练模型
for epoch in range(NUM_EPOCH):
    train_loss,train_acc = train(model,train_iterator,optimizer,crit)
    valid_loss,valid_acc = evaluate(model,valid_iterator,crit)

    if valid_acc>best_valid_acc:  #结果比最好的结果好
        best_valid_acc = valid_acc
        torch.save(model.state_dict(),'wordavg-model.pth')    #将参数保存到文件中



#######--------使用训练好的模型来做推理-----------
import scapy   #主要是用来做分词的:即把一个句子分成若干个单词
nlp = spacy.load('en')
def predict_sentiment(sentence):     #输入一个句子，预测这个句子是积极还是消极的
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]   #根据string找到index
    tensor = torch.LongTensor(indexed).to(device)   #将索引转换成张量
    tensor = tensor.unsqueeze(1)   #对张量增加一个维度
    pred = torch.sigmoid(model(tensor))  #完成对输入句子的分析
    return pred.item()
print(predict_sentiment('this is a terrible film'))






