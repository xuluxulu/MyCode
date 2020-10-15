#####----实现GAN的程序代码-----
from __future__ import division
from torchvision import models   #此models中自带很多定义好的模型
from torchvision import transfoms
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
#对图像进行转换：先转换成张量，然后在进行归一化
transfoms = transforms.Compose([
    transforms.ToTensor(),   #将数据转换成张量
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std = (0.5,0.5,0.5))
])

#获得mnist的训练数据,对数据集进行划分batch_size
mnist_data = torchvision.datasets.MNIST("./mnist_data",train= True,download = True,transform = transforms)
dataloader = torch.utils.data.DataLoader(dataset = mnist_data,
                                         batch_size =batch_size,
                                         shuffle = True)
#可视化图像
plt.imshow(next(iter(dataloader))[0][1][0])   #0表示第一批次，1表示第二张图像 ，显示的图像是彩色的
plt.imshow(next(iter(dataloader))[0][1][0],cmap = plt.cm.gray)   #显示的图像是黑白的

image_size = 28*28    #输入图像的尺寸
hidden_size = 256

#----------生成器的卷积过程和辨别器的卷积过程是相反的-----------
#-------discrimitor（辨别器）-------
D = nn.Sequential(
    nn.Linear(image_size,hidden_size),
    nn.LeakyReLU(0.2),    #采用leakrelu作为激活函数
    nn.Linear(hidden_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)

#------generator(生成器)：其图像的卷积过程和辨别器的卷积过程刚好相反------
latent_size = 64
G = nn.Sequential(
    nn.Linear(latent_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,image_size),
    nn.Tanh()   #将结果转换到-1到1之间
)

#将两个模型都放到gpu上
D = D.to(device)
G = G.to(device)

loss_fn = nn.BCELoss()   #多分类损失函数
d_optimizer = torch.optim.Adam(D.parameters(),lr = 0.002)    #定义辨别器的损失函数
g_optimizer = torch.optim.Adam(G.parameters(),lr = 0.002)    #定义生成器的损失函数

#开始训练模型：辨别器辨别生成的图像标签为0，真实图像为1，所以其损失是两个
# 生成器希望生成的图像的的标签为1
total_label = len(dataloader)
num_epochs = 30
for epoch in range(num_epochs):
    for i,(images,_) in enumerate(dataloader):

        batch_size = images.shape[0]
        #真实图像
        images = image.reshape(batch_size,image_size).to(device)   #重置图像的尺寸，并放到设备上

        #数据的标签
        real_labels = torch.ones(batch_size,1).to(device)  #真实图像的标签
        fake_labels = torch.zeros(batch_size,1).to(device)     #生成图像的标签

        outputs = D(images)  #辨别真实图像
        d_loss_real = loss_fn(outputs,real_labels)   #辨别器辨别真实图像的损失函数
        real_scores = outputs   #对于D，这个值越大越好

        #生成虚假图像，让辨别器去辨认
        z = torch.randn(batch_size,latent_size).to(device)   #一般都是根据噪声生成虚假图像
        fake_images = G(z)
        outputs = D(fake_images.detach())    #将虚假图像展开
        d_loss_fake = loss_fn(outputs,fake_labels)
        fake_score = outputs   #对于D，这个值越小越好

        #开始优化discriminator
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #开始优化generator  希望辨别器辨别虚假图像的标签为1
        outputs = D(fake_images)
        g_loss = loss_fn(outpus,real_labels)
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        print(epoch,d_loss.item(),g_loss.item())



