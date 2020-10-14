#----定义一个CNN网络实现MNIST数据集图像的卷积操作，实现分类---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #此处的卷积包括点乘和加上bias
        self.conv1 = nn.Conv2d(1,20,5,1)    #输入通道，输出通道，卷积核大小，卷积步长
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4*4*50,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4*4*50)   #reshape的作用
        #后面是全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)

# #获得一个数据集
# mnist_data = datasets.MNIST("./mnist_data",train=True,
#                             download = True,  #没有数据集则自动下载
#                             #将数据集转换成张量
#                             transforms = transforms.Compose([transforms.Tensor(),]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #获得设备
batch_size = 32
#获得训练数据集，并进行批次划分
train_loader = torch.utils.data.DataLoader(
          mnist_data = datasets.MNIST("./mnist_data",train=True,
                                download = True,  #没有数据集则自动下载
                                #将数据集转换成张量
                                transforms = transforms.Compose([
                                    transforms.Tensor(),
                                    transforms.Normalize((0.1307,),(0.3281,))  #对数据进行归一化，0.1307是均值，0.3281是方差
                                  ])),
          batch_size = batch_size,shuffle = True,
          num_worker = 1,pin_memory = True    #pin_memory可以加快训练速度
     )

#获得测试数据集，并进行批次划分
test_loader = torch.utils.data.DataLoader(
          mnist_data = datasets.MNIST("./mnist_data",train=False,   #train为False表示不是训练数据集，而是测试数据集
                                download = True,  #没有数据集则自动下载
                                #将数据集转换成张量
                                transforms = transforms.Compose([
                                    transforms.Tensor(),
                                    transforms.Normalize((0.1307,),(0.3281,))  #对数据进行归一化，0.1307是均值，0.3281是方差
                                  ])),
          batch_size = batch_size,shuffle = False,
          num_worker = 1,pin_memory = True    #pin_memory可以加快训练速度
     )

def train(model,device,train_loader,optimizer):
    model.train()
    for idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)   #将数据放到gpu上
        pred = model(data)
        loss = F.nll_loss(pred,target)

        #SGD损失函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx%100==0:
            print(idx,loss)   #打印出损失率


def test(model,device,test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():     #让下列的计算图不占用内存空间
        for idx,(data,target) in enumerate(test_loader):
            data,target = data.to(device),target.to(device)
            output = model(data)   #batch_size *10
            #总的损失率
            total_loss += F.nll_loss(output,target,reduction = 'sum').item()    #负对数似然损失


            pred = output.argmax(dim=1)  # batch_size*10，其中，dimension=1表示行，获得每一行最大值的索引
            #总的正确率
            correct +=pred.eq(target.view_as(pred)).sum().item()   #target.view_as().sum()是标签的维度转换为和pred一样的维度,然后判断二者是否相等
            if idx%100==0:
                print(idx,loss.item())

    total_loss /=len(test_loader.dataset)
    print('准确率为',total_loss)




lr = 0.01   #定义一个学习率
momentum = 0.5    #动量因子
model = Net().to(device)   #将模型放到gpu上
optimizer = torch.optim.SGD(model.parameters(),lr = lr,momentum = momentum)   #定义模型的优化器

num_epochs = 2
for epoch in range(num_epochs):
    train(model,device,train_loader,optimizer,epoch)
    test(model,device,test_loader)
torch.save(model.state_dict(),'cnn_model.pth')




