#####-----使用cnn实现迁移学习------
#即使用一个预训练的模型训练一个新的模型
import numpy as np
import torchvision
from torchvision import datasets,transforms,models   #此处的models中自带了很多模型
import matplotlib.pylplot as plt
import time
import os
import copy

#定义一些常量
data_dir = './hymenoptera_data'    #数据集的路径
model_name = 'resnet'    #保存模型的名称
num_classes = 2    #实现的是二分类的问题
batch_size = 32
num_epoch = 15
feature_extract = True   #即利用模型进行特征提取，不重新训练模型的参数，不改变模型的结构
input_size = 224
#
# #读入数据，并将其转换成相应的格式
# all_imgs = datasets.ImageFolder(os.path.join(data_dir,"train"),
#                                 #通过各种方式的操作初始化图像，并将其转换张量的形式
#                                 transforms.Compose([transforms.RandomResizedCrop(input_size),
#                                 transforms.RandomHorizontalFlip(),  #随便截取一个区域
#                                 transforms.ToTensor(),
#                                 ]))
# #对数据进行batch_size
# loader = torch.utils.data.DataLoader(all_imgs,batch_size= batch_size,shuffle = True,num_workers = 4)

#定义加载两个不同的数据集
data_transforms  = {   #.Compose()把所有的数据拼接起来
    "train":transforms.Compose([transform.RandomResizedCrop(input_size),  #随机的裁剪一块区域为input_size*input_size
                                transform.RandomHorizontalFlip(),   #进行水平随机翻转
                                transform.ToTensor(),  #将图像转换成Tensor
                                #定义一系列的均值和方差
                                transform.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    "val":transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size,   #以图像中心位置进行裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
      ])}

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])  for x in ['train','val']}

#此处得到所需的数据
dataloaders_dict = {x:torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size = batch_size,
                                                  shuffle = True,
                                                  num_worker =4) for x in ['train','val']}

#设定使用gpu还是cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#此处定义的函数作用：判断使用已有模型来提取特征（即保留原模型的结构和参数，feature_extract = True），还是对模型结构进行改变（即重新训练模型, fearure_extract = False）
def set_parameter_requires_grad(model,feature_extract):
    if feature_extract:   #此处若为True，表示则用模型来提取特征，则原模型的参数则不进行改变
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name,feature_extract,num_classes,use_pretrained = True):
    if model_name=="resnet":
        model_ft = models.resnet18(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)   #对模型的处理方式
        num_ftrs = model_ft.fc.in_features   #得到全连接模型的输入特征
        #此处更改model_ft的最后一层全连接层
        model_ft.fc = nn.Linear(num_ftrs,num_classes)    #自定义一个线性转换
        input_size = 224
    else:
        return None,None
    return model_ft,input_size

#初始化模型参数，即只有最后一层的linear的参数会被训练
model_ft,input_size = initialize_model(model_name,feature_extract,
                                       num_classes,use_pretrained = True)


#定义训练模型
def train_model(model,dataloaders,loss_fn,optimizer,num_epochs = 5):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.

    for epoch in range(num_epochs):
        for phase in ['train','val']:
            running_loss = 0.0
            running_acc = 0.0
            valid_acc_history =[]
            if phase =='train':
                model.train()
            else:
                model.eval()
            for inputs,labels in dataloaders_dict(phase):
                inputs,labels = inputs.to(device),labels.to(device)   #将输入的各种数据都放到gpu上
                with torch.autograd.set_grad_enabled(phase =='train'): #表示后续的参数的grad都不用更新
                    outputs = model(inputs)
                    loss = loss_fn(outputs,labels)
                preds = outputs.argmax(dim = 1)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #计算准确率和错误率
                running_loss  += loss.item() *inputs.size(0)  #因为loss都是求过均值的
                running_acc +=torch.sum(preds.view(-1)==labels.view(-1).item())

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_acc/len(dataloaders[phase].dataset)

            print('损失函数：',epoch_loss,'准确率:',epoch_acc)

            #将最好的权重保存起来
            if phase =='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            if phase=='val':
                valid_acc_history.append(epoch_acc)


    model.load_state_dict(best_model_weights)   #加载最好的模型

    return model,valid_acc_history

model_ft = model_ft.to(device)   #把模型放到gpu上
optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,
                                   model_ft.parameters()),lr = 0.001,momentum = 0.9)
loss_fn = nn.CrossEntropyLoss()


#开始训练模型
train_model(model_ft,dataloaders_dict,loss_fn,optimizer,num_epochs =num_epochs)




# #-----以下是对模型进行可视化操作-----
# unloader = transforms.ToPILImage()   #将图像转换成pil格式
# plt.ion()   #动态展示图片
# def imshow(tensor,title = None):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)  #将第一维度删除
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)   #给图像加上标签
#     plt.pause(0.001)
# plt.figure()
# imshow(img[31],title='Image')   #调用上面定义的函数



