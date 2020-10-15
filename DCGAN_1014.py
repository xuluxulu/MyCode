###---DCGAN:基于反卷积的GAN操作-------
import torchvision.utils as vutils

#加载各种数据集
image_size = 64
batch_size = 128
dataroot = 'celeba/img_align_celeba'   #数据集的根目录
num_workers = 2

dataset = torchvision.datasets.ImageFolder(root = dataroot,transform = transform.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),  #以中心位置进行裁剪
    transforms.ToTensor(),    #将数据转换成Tensor
    transforms.Nomalize((0.5,0.5,0.5),(0.5,0.5,0.5))   #数据归一化，加快模型的训练速度
]))

dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = num_workers)


#初始化一系列参数
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:   #卷积的初始方式
        nn.init.normal_(m.weight.data,0.0,0.02)   #均值为0.0，方差为0.02
    elif classname.find("BatchNorm") !=-1:   #批量归一个的初始化方式
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)   #bias初始化为0


##生成器Generator
nz = 100   #输入通道
ngf = 64   #输出通道
nc = 3

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            #反卷积操作的参数：in_channels,out_channels,kernel_size,stride =1,padding = 0,output_padding = 0,groups = 1,bias = True,dilation = 1
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf,nc,4,2,1,bias = False),    #最后变成3个通道的图像
            nn.Tanh()
        )
    def forward(self,x):
        return self.main(x)
G = Generator().to(device)
G.apply(weight_init)   #初始化G中的一系列参数


ndf = 64   #输入通道的数量
class Discriminator(nn.Module):
    def __init__(self):
        supre(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias= True),
            nn.LeakyRelu(0.2,inplace = True),

            nn.Conv2d(ndf,ndf*2,4,2,1,bias = True),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyRelu(0.2,inplace = True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 24),
            nn.LeakyRelu(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyRelu(0.2, inplace=True),

            nn.Conv2d(ndf*8,1,4,1,bias = Flase),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x)

D = Discriminator().to(device)
D.apply(weight_init)

#定义损失函数和两个优化器
loss_fn = nn.BCELoss()
d_optimizer = nn.optim.Adam(D.parameters(),lr = 0.002,betas = [0.5,0.999])
g_optimizer = nn.optim.Adam(G.parameters(),lr = 0.002,betas = [0.5,0.999])


total_steps = len(dataloader)
num_epochs = 30
for epoch in range(num_epochs):
    for i,data in enumerate(dataloader):

        D.zero_grad()
        real_images = data[0].to(device)   #将真实图像放到设备上
        b_size = real_images.size(0)

        label = torch.ones(b_size,1).to(device)
        output = D(real_images).view(-1)

        real_loss = loss_fn(output,label)
        real_loss.backward()
        D_x = output.mean().item()

        #生成图像
        noise = torch.randn(b_size,nz,1,1,device =device)
        fake_images = G(noise)
        label.fill_(0)
        fake_labels = torch.zeros(b_size,1).to(device)
        output= D(fake_images.detach().view(-1))
        fake_loss = loss_fn(output,fake_labels)
        D_G_z1 = output.mean().item()
        loss_D = real_loss + fake_loss
        loss_D.backward()
        d_optimizer.step()


        #训练generator
        G.zero_grad()
        label.fill_(1)
        output = D(fake_images.view(-1))
        loss_G = loss_fn(output,label)
        loss_G.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()

        if i%200==0:
            print(i,loss_D.item(),loss_G.item())   #显示各种准确率
