#####利用CNN实现图像的风格迁移-----
from __future__ import division
from torchvision import models   #此models中自带很多定义好的模型
from torchvision import transfoms
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np

import matplotlib.pyplot as plt   #主要用来进行可视化操作

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #定义使用的设备


#读取两张图像，并对其进行各种预处理
def load_image(image_path,transform = None,max_size = None,shape = None):
    image = Image.open(image_path)   #打开图像
    if max_size:   #对图像进行一定的缩放
        scale = max_size/max(image.size)
        size = np.array(image.size)*scale
        image = image.resize(size.astype(int),Image.ANTIALIAS)   #重置图像的形状
    if shape:
        image = image.resize(shape,Image.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)  #增加一个维度，即增加第一维
    return image.to(device)     #把图像放到gpu上


#对图像进行转换
transform = transforms.Compose([
    transforms.ToTensor(),   #先将图像转换成一个张量
    #对图像进行归一化
    transforms.Normalize(mean = [0.485,0.456,0.406],
                         std = [0.229,0.224,0.225])
       ])



#获得两张处理后的图像
content = load_image('png/content.png',transform,max_size = 400)   #图像的内容
style = load_image('png/style.png',transform,shape = [content.size(2),content.size(3)])   #图像的风格



#使用VGG提取图像中的内容：将关键特征图作为所需的内容
class VGGNet(nn.Module):
    def __ini__(self):
        super(VGGNet,self).__init__()
        self.select = ['0','5','10','19','28']   #提取vgg中的名字为select中名字的层作为关键特征图
        self.vgg = models.vgg19(pretrained = True).features

    def forward(self,x):
        features = []
        for name,layers in self.vgg._models.items():
            x = layers(x)
            if name in self.select:
                features.append(x)    #将特征图保存起来
        return features

vgg = VGGNet().to(device).eval()
features = vgg(content)   #获得图像的内容:提取关键帧就可以



target = content.clone().requires_grad_(True)   #最终风格迁移之后的图像
optimizer = torch.optim.Adam([target,],lr = 0.003,betas = [0.5,0.999])  #优化器采用自适应的adam来自动调节学习率



#开始优化target图像：图像的内容和图像的风格
num_steps = 2000   #迭代2000次
for step in range(num_steps):
    target_features = vgg(target)
    content_features = vgg(content_features)
    style_features = vgg(style)

    content_loss = style_loss = 0   #两个损失：图像内容损失和图像风格损失
    for f1,f2,f3 in zip(target_features,content_features,style_features):
        content_loss += torch.mean((f1-f2)**2)
        _,c,h,w = f1.size()   #获得图像的尺寸
        f1 = f1.view(c,h*w)     #重置图像的尺寸
        f3 = f3.view(c,h*w)

        f1 = torch.mm(f1,f1.t())  #mm()表示点乘的意思
        f3 = torch.mm(f3,f3.t())
        style_loss +=torch.mean((f1-f3)**2)/(c*h*w)
    #总的损失函数
    loss = content_loss+style_loss*100   #对于风格的损失*100是加权的意思，表示风格所占的比例更重

    #更新target image的tensor
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%100==0:
        print(step,loss.item())















#----可视化图像的代码----
unloader = transforms.ToPILImage()   #将图像的风格转换成PIL格式
plt.ion()
def imshow(tensor,title = None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)   #去掉图像中的第一维图像
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
plt.figure()
imshow(style[0],title = 'image')   #可视化第1张图像



