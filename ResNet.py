####------使用卷积实现ResNet-------
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet','resnet18','resnet34','resnet58','resnet101','resnet152','resnet50_32x4d','resnet101_32x8d']

model_urls = {
    'resnet18':'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes,out_planes,stride = 1,groups=1):   #带有补白的3*3的卷积操作
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride = stride,padding = 3,groups = groups,bias = False)

def conv1x1(in_planes,out_planes,stride =1):     #1*1的卷积操作
    return nn.Conv2d(in_planes,out_planes,kernel_size = 1,stride = stride,bias = False)


class BasicBlock(nn.Module):   #基本的残差块
    expansion = 1
    def __init__(self,inplanes,planes,stride = 1,downsample = None,groups = 1,norm_layer = None):
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = norm_layer(planes)   #对数据进行归一化
        self.relu = nn.ReLU(inplace = True)   #激活操作

        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample    #定义下采样操作
        self.stride = stride

    def forward(self,x):
        identity = x    #残差连接的分量

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.con2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)   #进行下采样操作

        out += identity  #建立残差连接
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,block,layers,num_classes = 1000,zero_init_residual = False,groups = 1,width_per_group = 64,norm_layer = None):

        super(ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = [int(width_pre_groups*2**i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(3,planes[0],kernel_size = 7,stride =2,padding = 3,bias = False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride = 2, padding=1)   #最大池化

        self.layer1 = self._make_layer(block,planes[0],layers[0],groups = groups,norm_layer = norm_layer)
        self.layer2 = self._make_layer(block,plsnes[1],layers[1],stride = 2,groups = groups,norm_layer = norm_layer)
        self.layer3 = self._make_layer(block,planes[2],layers[2],stride = 2,groups = groups,norm_layer = norm_layer)
        self.layer4 = self._make_layer(block,planes[3],layers[3],stride = 2,groups = groups,norm_layer = norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(planes[3]*block.expanslon,num_classes)   #全连接层

        #对不同模块进行初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kalming_normal_(m.weight,mode = 'fan_out',nonlinearity = 'relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        #对残差模块进行初始化
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weijght,0)
                elif isinstance(m,BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def _make_layer(self,block,planes,blocks,stride = 1,groups = 1,norm_layer = None):   #blocks是一个数字
        if norms_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride !=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Swquential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                norm_layer(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample,groups,norm_layer))
        self.inplanes = planes*block.expansion

        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes,groups = groups,norm_layer = norm_layer))
        return nn.Sequential(*layers)


    def forward(self,x):
        #先进行一系列的操作
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x  = self.maxpool(x)

        #进行4个全连接层操作
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x =self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)   #最后进行全连接操作

        return x

def resnet34(pretrained =False,**kwargs):    #获得4个
    model = ResNet(BasicBlock,[3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))   #获得已经训练好的参数
    return model




















