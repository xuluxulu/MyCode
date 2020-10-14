# ##------------用numpy实现两层神经网络------
# #一个全连接Relu神经网络，一个隐含层，没有Bias,用来从x预测到y,使用L2 loss
# #hidden = wx+b
# #a  = max(0,hidden)   激活函数
# #yout = w2*a + b2
# import numpy as np
# N,D_in,H,D_out = 64,1000,100,10  #训练的数据量，输入维度，隐藏层，输出维度
# #随机创建一些训练数据
# x = np.random.randn(N,D_in)
# y = np.random.randn(N,D_out)
#
# #权重
# w1 = np.random.randn(D_in,H)
# w2 = np.random.randn(H,D_out)
#
# #定义一个学习率
# learning_rate = 1e-6    #10的-6次方
# for t in range(500):
#     #forward pass
#     h = x.dot(w1)   #x*w1
#     h_relu = np.maximum(0,h)   #relu激活函数
#     y_pred = h_relu.dot(w2)
#
#     #compute loss
#     loss = np.square(y_pred - y).sum()   #均方误差求和
#     print(t,loss)
#
#     #compute gradient  aa.T指的是对aa进行转置
#     grad_y_pred = 2.0*(y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()   #grad_h和上述的grad_h_relu一样
#     grad_h[h<0] = 0
#     grad_w1 = x.T.dot(grad_h)
#
#     #update weights of w1 and w2
#     w1 -=learning_rate*grad_w1
#     w2 -=learning_rate*grad_w2


# ####------------用tensor实现两层神经网络------
# #一个全连接Relu神经网络，一个隐含层，没有Bias,用来从x预测到y,使用L2 loss
# #hidden = wx+b
# #a  = max(0,hidden)   激活函数
# #yout = w2*a + b2
# import torch
# N,D_in,H,D_out = 64,1000,100,10  #训练的数据量，输入维度，隐藏层，输出维度
# #随机创建一些训练数据
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
#
# #权重
# w1 = torch.randn(D_in,H)
# w2 = torch.randn(H,D_out)
#
# #定义一个学习率
# learning_rate = 1e-6    #10的-6次方
# for t in range(500):
#     #forward pass
#     h = x.mm(w1)   #x*w1
#     h_relu = h.clamp(min = 0)   #relu激活函数:让小于0的数转换成0，
#     y_pred = h_relu.mm(w2)
#
#     #compute loss
#     loss = (y_pred - y).pow(2).sum()   #均方误差求和
#     print(t,loss.item())
#
#     #compute gradient  aa.T指的是对aa进行转置
#     grad_y_pred = 2.0*(y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)    #grad_w2.grad   直接就可获得梯度
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()   #grad_h和上述的grad_h_relu一样
#     grad_h[h<0] = 0        #grad_h.grad
#     grad_w1 = x.t().mm(grad_h)     #grad_w1.grad、
#
#
#     #update weights of w1 and w2
#     w1 -=learning_rate*grad_w1
#     w2 -=learning_rate*grad_w2



####------------用tensor实现两层神经网络:使用其自带函数（backward）------
#一个全连接Relu神经网络，一个隐含层，没有Bias,用来从x预测到y,使用L2 loss
#hidden = wx+b
#a  = max(0,hidden)   激活函数
#yout = w2*a + b2
import torch
N,D_in,H,D_out = 64,1000,100,10  #训练的数据量，输入维度，隐藏层，输出维度
#随机创建一些训练数据
x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

#权重
w1 = torch.randn(D_in,H,requires_grad = True)  #此处requires_grad = True的意思是要求分配内存给梯度
w2 = torch.randn(H,D_out,requires_grad = True)

#定义一个学习率
learning_rate = 1e-6    #10的-6次方
for t in range(500):
    #forward pass
    h = x.mm(w1)   #x*w1
    h_relu = h.clamp(min = 0)   #relu激活函数:让小于0的数转换成0，
    y_pred = h_relu.mm(w2)

    #compute loss
    loss = (y_pred - y).pow(2).sum()   #均方误差求和
    print(t,loss.item())

    loss.backward()   #反向传播
    with torch.no_grad():  #因为下述的w1和w2也是计算图，为了不让计算图占内存，所以使用此行命令
        w1 -= learning_rate *w1.grad
        w2 -= learning_rate *w2.grad
        w1.grad.zero_()   #把每个w1.grad清零，否则grad会一直叠加
        w2.grad.zero_()





####------------用tensor实现两层神经网络:torch.nn------
#一个全连接Relu神经网络，一个隐含层，没有Bias,用来从x预测到y,使用L2 loss
#hidden = wx+b
#a  = max(0,hidden)   激活函数
#yout = w2*a + b2
import torch.nn as nn
N,D_in,H,D_out = 64,1000,100,10  #训练的数据量，输入维度，隐藏层，输出维度
#随机创建一些训练数据
x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

model = nn.Sequential(
       nn.Linear(D_in,H),   #wx+b ,默认有bias的,即nn.Linear(D_in,H,Bias = True)
       nn.ReLU(),
       nn.Linear(H,D_out),
)
#以normal的形式重新初始化模型中的两个线性模型
nn.init.normal_(model[0].weight)
nn.init.normal_(model[2].weight)
#model = model.cuda()    #将模型放到gpu上运行
loss_fn = nn.MSELoss(reduction = 'sum')    #均方误差，并求和
#定义一个学习率
learning_rate = 1e-6    #10的-6次方
for t in range(500):
    #forward pass
    y_pred = model(x)
    #compute loss
    loss =loss_fn(y_pred,y)  #均方误差求和
    print(t,loss.item())

    loss.backward()   #反向传播

    #update weights
    with torch.no_grad():
        for param in model.parameters():   #param{tensor,grad}
            param -= learning_rate *param.grad   #对里面的参数进行更新
    model.zero_grad()  #，对grad清空，防止grad叠加





