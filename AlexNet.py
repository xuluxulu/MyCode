####------使用CNN实现AlexNet网络结构----
import torch.nn as nn
import torch.utils.model_zoo  as model_zoo

__all__ = ['AlexNet','alexnet']
#加载已经训练好的模型参数地址
model_urls = {
     'alexnet':'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

#定义模型的网络结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        #定义一系列的卷积
        self.feature = nn.Sequence(
            nn.Conv2d(3,64,kernel_size = 11,stride = 4,padding = 2),  #输入通道、输出通道
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3,stride = 2),

            nn.Conv2d(64,192,kernel_size = 5,stride =2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size =3,stride = 2),

            nn.Conv2d(192,384,kernel_size=3,stride = 1),
            nn.ReLU(inplace = True),

            nn.Conv2d(384,256,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),

            nn.Conv2d(256,256,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3,stride = 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))   #平均池化

        #分类操作：全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096,num_classes),
        )

    def forward(self,x):
        x = self.featurs(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),256*6*6)   #x.view()相当于reshape()操作
        x = self.classifier(x)
        return x
def alexnet(pretrained = False,**kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model.zoo.load_url(model_urls['alexnet']))   #加载已有的模型
    return model