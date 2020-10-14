#------使用CNN（体现局部的特征）实现文本分类---------
#原理：1.首先对每个单词进行embed，得到每个单词的向量[1,k]
#2.把一个句子中所有单词n的向量组合成一个矩阵[n,k]（之后在这个矩阵上做卷积操作）
#3.卷积核大小为[h,k],其中h小于n的，卷积核的个数为w个
#4.进行卷积操作：1）只有一个卷积层，就和图像卷积一样，每一卷积核得到一个值（点乘），在此处使用w个卷积核就可以得到一个单词向量，单词的数量是n-h+1,每个单词向量的长度为w
#             2)若有多个卷积层（如3个），则对这3个卷积层相同位置的结果，取其中的最大值做为该卷积核的结果，其他和一个卷积核的操作一样
#5. 进行dropout和线性变换，进行文本分析预测
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #设定运行设备

#对不同测试进行批次划分:BucketIterator是把长度差不多的句子的长度划分到一个batch中，确保每一个batch中不会出现太多的padding,并自动打乱句子（此句子是完整的+label）
train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size = BATCH_SIZE,
    device = device
     )


VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)   #pad_token得到的是pad的对应的索引
class CNN(nn.Module):
    def __init__(self,vocab_size,embedding_size,output_size,pad_idx,num_filters,filter_size):
        super(CNN,self).__init__()
        self.embed = nn.embedding(vocab_size,embedding_size,padding_idx = pad_idx)
        #self.conv = nn.Conv2d(in_channels = 1,out_channels = num_filters,kernel_size = (filter_size,embedding_size))
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels = 1,out_channels = num_filters,kernel_size=(fs,embeddign_size)) for fs in filter_sizes]

        )
        self.linear = nn.Linear(embedding_size*len(filter_sizes),output_size)
        self.dropout = nn.Dropout.dropout()

    def forward(self,text):   #最初只定义一个卷积，后来定义一个卷积序列，即3个卷积
        text = text.permute(1,0)   #交换维度为1和0的维度  [batch_size,seq_len]
        embedding = self.embed(text)    #对单词进行向量的预测   [batch_size,seq_len,embedding_size]
        embedding = embedding.unsqueeze(1)  #增加一个第二维,1表示索引，索引从0开始，表示第一维度  [batch_size,1,seq_len,embeddign_size]
        #conved = F.relu(self.conv(embedding))   #对向量进行卷积操作   [batch_size,num_filters,seq_len-filter_size +1,1]
        # conved = conved.squeeze(3)   #把第4维度去掉   [batch_size,num_filters,seq_len-filter_size+1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]   #进行多次卷积操作
        # pooled = F.max_pooled(conved,conved.shape[2])    #[batch_size,num_filter,1]
        # pooled = pooled.squeeze(2)
        # pooled = self.dropout(pooled)
        # pooled = self.dropout(pooled)
        # return self.linear(pooled)
        #进行3次池化
        pooled = [F.max_pooled(conv,conv.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.concat(pooled,dim = 1)  #对所有的pooled按第2维度进行拼接
        pooled = self.dropout(pooled)
        return self.linear(pooled)


#初始化模型
model = CNN(vocab_size= VOCAB_SZIE,
            embedding_size = EMBEDDING_SIZE,
            output_size = OUTPUT_SIZE,
            pad_idx = PAD_IDX,
            num_filters = 100,
           # filter_size = 3,
            filter_sizes = [3,5,3],
            dropout = 0.5)


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
        preds = model(batch.text).squeeze()   #去掉第一个维度
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
        torch.save(model.state_dict(),'cnn-model.pth')    #将参数保存到文件中


#---测试模型--
model.load_state_dict('cnn-load.pth')
test_loss,test_acc = evaluate(model,test_iterator,crit)
print('准确率',test_acc)