import numpy as np
import torch
NUM_DIGITS = 10

def fizz_buzz_encode(i):
    if i%15 ==0:
        return 3
    elif i%5==0:
        return 2
    elif i%3==0:
        return 1
    else:
        return 0
def fizz_buzz_decode(i,prediction):
    return [str[i],'fizz','buzz','fizzbuzz'][prediction]

def binary_encode(i,num_digits):   #将数据转换成二进制
    return np.array([i >>d & 1 for d in range(num_digits)][::-1])


#训练数据是101到1000
trX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])

NUM_HIDDEN = 100
#定义模型
model = torch.nn.Squential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN,4),
)

if torch.cuda.is_available():
    model = model.cuda()  #将模型放到gpu上

loss_fn = torch.nn.CrossEntropyLoss()   #交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(),lr = 0.05)
BATCH_SIZE = 128

#----进行模型训练
for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):   #一个batch一个batch的取数据
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        if torch.cuda.is_available():   #若cuda存在，则把两个张量放到gpu上
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#------模型预测
#把1到100的数据作为测试数据
testX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(i,101)])
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(0,101),testY.max(1)[1].cpu().data.tolist())
print([fizz_buzz_encode(i,x) for i,x in predictions])