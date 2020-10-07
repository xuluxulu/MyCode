#def add(x,y):
#    aa = x+y
#    print(aa)  

########函数讲解链接：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-rnix2gv7.html
#######Tensorflow搭建的神经网络############################



########输出每一次训练的参数
#import tensorflow as tf
#import numpy as np
#
#
##创建数据
#x_data = np.random.rand(100).astype(np.float32)   #随机生成一个包含100个数据的数组，其每个数据都是处在0到1之间
##print(type(x_data))
##print(x_data)
#y_data= x_data*0.1 + 0.3
#
#
##创建tensorflow的初始参数
#Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))    #随机产生一个一维的数据，该数据只有一个，而且该数据的范围处于-1到1之间
#biases = tf.Variable(tf.zeros([1]))   #随机产生一个数据，该数据给初始值为0
#
#
##定义网路结构
#y = Weights*x_data + biases
#loss = tf.reduce_mean(tf.square(y-y_data))  #先求查，再求均值
#optimizer = tf.train.GradientDescentOptimizer(0.5)    #使用随机梯度下降的方法，学习率为0.5
#train = optimizer.minimize(loss)  #最小化损失函数
#
#init = tf.initialize_all_variables()  #初始化参数
#sess = tf.Session()
#sess.run(init)   #初始化参数，只有run()才能把以上的所定义的各种流程都给激活了，网路真正的开始运行起来
#
#for step in range(201):
#    sess.run(train)   #开始训练
#    if step%20 ==0:  #每隔20次将迭代的参数输出来：权值和阈值
#        print(step,sess.run(Weights),sess.run(biases))
#        #最终观察到Weights更加接近0.1，biases更加接近0.3
        


######Session会话控制
#import tensorflow as tf
#
##定义两个常量矩阵
#matrix1 = tf.constant([[3,3]])  #该矩阵是一个一行两列的矩阵，其值都是3 
#matrix2 = tf.constant([[4],
#                       [4]])   #该矩阵是一个两行一列的矩阵，其值都是2
#
#product = tf.matmul(matrix1,matrix2)    #两个矩阵相乘   np.dot(m1,m2)  
#
##方法一
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()
#
#
##方法二 ：其方式不用去关闭sess():此方式常用
#with tf.Session() as sess:
#    result2 = sess.run(product)
#    print(result2)



######variable变量的用法
#import tensorflow as tf
#state = tf.Variable(0,name ='counter')  #声明一个变量，该变量初始值为0，名字为‘counter’
#print(state)
#one = tf.constant(1)         #定义一个常量
#
#new_value = tf.add(state,one)      #将两个值进行相加
#update = tf.assign(state,new_value)    #将new_value的值赋给state
#
#init = tf.initialize_all_variables()   #由于上述定义了变量，所以此处一定要使用该函数进行初始化
#
#with tf.Session() as sess:
#    sess.run(init)       #初始化参数
#    for _ in range(3):   
#        sess.run(update)       #运行赋值工作
#        print(sess.run(state))      #运行得到的结果
        


########placeholder传入值的使用
#import tensorflow as tf
#
##input1 = tf.placeholder(tf.float32,[2,2])    #定义input1中的数据类型是tf.float32,其结构是两行两列
#input1 = tf.placeholder(tf.float32)
#input2 = tf.placeholder(tf.float32)
#
#output = tf.matmul(input1,input2)
#
#with tf.Session() as sess:
#    print(sess.run(output,feed_dict ={input1:[7.],input2:[3.]}))   #以字典形式给其赋值
    

########激励函数、添加层、可视化数据
#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#
#def add_layer(inputs,in_size,out_size,activation_function = None):
#    Weights = tf.Variable(tf.random_normal([in_size,out_size]))   #随机生成in_size行out_size列的元素（数据处在0到1之间）
#    #tf.zeros()是生成1行out_size列，元素全是0
#    biases = tf.Variable(tf.zeros([1,out_size])) +0.1
#    Wx_plus_b = tf.matmul(inputs,Weights) + biases
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b)
#    return outputs
#    
#
##定义数据
#x_data= np.linspace(-1,1,300)[:,np.newaxis]   #生成300行1列的数据（处在-1到1之间）
#noise = np.random.normal(0,0.05,x_data.shape)   #产生噪声点，方差为0.05，shape与x_data一样
#y_data = np.square(x_data)-0.5 +noise
#
##申明两个占位符
#xs = tf.placeholder(tf.float32,[None,1])  #输入值为一个数据
#ys = tf.placeholder(tf.float32,[None,1]) #输出值也是一个数据
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data,y_data)  #添加可视化数据
#plt.ion()   #不会暂停，连续的生成一个图
#plt.show()  #将图显示出来
#
#
#l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
#prediction = add_layer(l1,10,1,activation_function = None)
#
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                                    reduction_indices = [1]))
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
##init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)  #初始化所有参数
#
#for i in range(1000):
#    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#    if i%50==0:  #每50步输出一次损失函数
#        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#        try:
#            ax.lines.remove(lines[0])
#        except Exception:
#            pass
#        prediction_value = sess.run(prediction,feed_dict = {xs:x_data}) #获得预测的结果
#        lines = ax.plot(x_data,prediction_value,'r-',lw =5)  #画出一条红色的线
#        ax.lines.remove(lines[0])   #删除lines中的第一个线
#        plt.pause(0.1)   #暂停0.1秒再继续



##########tensorflow中的tensorboard的使用
#import tensorflow as tf
#import numpy as np
#def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
#    layer_name = 'layer%s'%n_layer
#    with tf.name_scope('layer'):
#        with tf.name_scope('weights'):    
#            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
#            tf.summary.histogram(layer_name+'weights',Weights)
#        with tf.name_scope('biases'):
#            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name = 'b')
#            tf.summary.histogram(layer_name+'biases',biases)
#        with tf.name_scope('Wx_plus_b'):
#            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
#        
#        if activation_function is None:
#            outputs = Wx_plus_b
#        else:
#            outputs = activation_function(Wx_plus_b)
#        tf.summary.histogram(layer_name+'outputs',outputs)
#        return outputs
#
##定义输入数据
#x_data = np.linspace(-1,1,300)[:,np.newaxis]
#noise = np.random.normal(0,0.05,x_data.shape) 
#y_data = np.square(x_data)-0.5 + noise   
#    
#    
##为输入变量定义两个placeholder
#with tf.name_scope('input'):
#    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
#    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')
#
#
##定义网络结构
#l1 = add_layer(xs,1,10,n_layer =1,activation_function = tf.nn.relu)
#
#prediction = add_layer(l1,10,1,n_layer =2,activation_function = None)
#
##计算损失进行梯度下降来优化参数
#with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                     reduction_indices =[1]))
#    tf.summary.scalar('loss',loss)
#with tf.name_scope('train'):
#    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
#
#sess = tf.Session()
##将上述使用的所有的summary的图进行合并
#merged = tf.summary.merge_all()
##把整个框架定义（写）到指定的文件夹中
#writer = tf.summary.FileWriter("logs/",sess.graph)
#
#sess.run(tf.initialize_all_variables())
#
#for i in range(1000):
#    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#    
#    ##每隔50步记录一个点
#    if i%50==0:
#        result = sess.run(merged,feed_dict = {xs:x_data,ys:y_data})
#        writer.add_summary(result,i)








#############对数据进行分类学习##############
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#
###手写字体识别----数字识别：第一次运行时，若没有这个数据包则会自动下载，第二次运行时则会自动加载
#mnist = input_data.read_data_sets('MNIST_data',one_dot = True)
#
#def add_layer(inputs,in_size,out_size,activation_function = None):
#    Weights = tf.Variable(tf.random.normal([in_size,out_size]))
#    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
#    Wx_plus_b = tf.matmul(input,Weights) + biases
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b)
#    return outputs
#
#def compute_accuracy(v_xs,v_ys):
#    global prediction
#    y_pre = sess.run(prediction,feed_dict={xs:v_xs})  #得到验证数据的验证结果标签：热码
#    
#    #比较预测值与真实值之间的关系
#    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))  #返回值是0或者1
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #先类型转换，再计算均值
#    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})  #得到一个百分比
#    return result
#
#
#
#
###定义placeholder
#xs = tf.placeholder(tf.float32,[None,784])   #输入：图片的大小28*28
#ys = tf.placeholder(tf.float32,[None,10])   #输出类别是10个类别:是热图的形式
#
###添加层
#prediction = add_layer(xs,784,10,activation_function = tf.nn.softmax)
#
##计算交叉熵损失
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices =[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#
#sess = tf.Session()
#
#sess.run(tf.initialize_all_variables())
#
#
##循环迭代
#for i in range(1000):
#    batch_xs,batch_ys = mnist.train.next_batch(100)  #提取出一部分数据
#    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#    
#    #训练数据每训练50次，就对测试数据进行测试一次
#    if i%50==0:
#        print(compute_accuracy(mnist.test.images,mnist.test.labels))  #用测试数据进行测试






######如何防止数据过拟合？Dropout进行防止：0到9的数据进行分类
#import tensorflow as tf
#from sklearn.datasets import load_digits
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import LabelBinarizer
#
#
##加载数据
#digits = load_digits()
#X = digits.data  #数据
#y = digits.target  #标签
#y = LabelBinarizer().fit_transform(y)  #把标签转换成热码
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 3)   #对数据集进行划分
#
##定义添加层
#def add_layer(inputs,in_size,out_size,layer_name,activation_function = None):
#    
#    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
#    Wx_plus_b = tf.matmul(inputs,Weights) + biases
#    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)  #对输出的结果进行dropout
#    
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b)
#    tf.summary.histogram(layer_name+'outputs',outputs)
#    return outputs
#
#keep_prob = tf.placeholder(tf.float32)   #定义保存的数据不被dropout
#xs = tf.placeholder(tf.float32,[None,64])   #图片的大小是32*32
#ys = tf.placeholder(tf.float32,[None,10])  #标签的类别是10个
#
#
#l1 = add_layer(xs,64,50,'l1',activation_function = tf.nn.tanh)
#prediction = add_layer(l1,50,10,'l2',activation_function = tf.nn.softmax)
#
##使用交叉熵损失函数计算损失
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices = [1]))
#tf.summary.scalar('loss',cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
#
#sess = tf.Session()
#merged = tf.summary.merge_all()
#
#train_writer = tf.summary.FileWriter('logs/train',sess.graph)
#test_writer = tf.summary.FileWriter('logs/test',sess.graph)
#
#sess.run(tf.initialize_all_variables())  #初始化所有的参数
#
#
#for i in  range(500):
#    sess.run(train_step,feed_dict = {xs:X_train,ys:y_train,keep_prob:0.6})  #保存60%的数据
#    
#    if i%50==0:  #每训练50次，对测试数据进行测试,并对测试数据和验证数据的结果进行保存
#        train_result = sess.run(merged,feed_dict = {xs:X_train,ys:y_train,keep_prob:1})  #值为1表示不被dropout
#        test_result = sess.run(merged,feed_dict ={xs:X_test,ys:y_test,keep_prob:1})
#        
#        
#        #把信息添加到文件中
#        train_writer.add_summary(train_result,i)
#        test_writer.add_summary(test_result,i)
        


#########卷积神经网络CNN#################
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow as tf
#
##数据是1到10
#mnist = input_data.read_data_sets('MNIST_data',one_hot = True)  #手写字体数据库
#
#
##为网络定义输入的placeholder
#xs = tf.placeholder(tf.float32,[None,784])  #输入图片的大小为28x28    
#ys = tf.placeholder(tf.float32,[None,10])    #输出数据的类别是10个
#keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs,[-1,28,28,-1])   #重新设置图片的形状为28*28
##print(x_image.shape)  #[n_samples,28,28,1]  n_samples是图片的形状，其中1表示该图像为黑白
#
#
#def compute_accuracy(v_xs,v_ys):
#    global prediction
#    y_pre = sess.run(prediction,feed_dict = {xs:v_xs,keep_prob:1})   #对测试数据进行预测结果，得到的类别是热码形式
#    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))   #预测结果与真实结果进行对比
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))     #进行数据类型转换
#    result = sess.run(accuracy,feed_dict = {xs:v_xs,ys:v_xs,keep_prob:1})  #得到的结果是一个百分比
#    return result
#
#
#
#def weight_variable(shape):  #shape表示图像的形状
#    initial = tf.truncated_normal(shape,stddev=0.1)  #随机产生数据的标准差为0.1，形状为shape
#    return tf.Variable(initial)
#
#
#def bias_variable(shape):
#    initial = tf.constant(0.1,shape =shape)  #产生数据的形状为shape，所有数据的都是0.1
#    return tf.Variable(initial)
#
##卷积层
#def conv2d(x,W):  #W有点类似于卷积核
#    #stride[1,x_movement,y_movement,1]
#    #Must have strides[0]=strides[3] = 1
#    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  #此处SAME表示长宽一样
#
##池化
#def max_pool_2x2(x):
#    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')  #此处是SAME表示通道一样
#
##定义卷积层1
#W_conv1 = weight_variable([5,5,1,32])  #5*5是卷积核的大小，1是输入数据的通道，32是输出数据的通道
#b_conv1 = bias_variable([32])
#h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)  #输出形状为28*28*32
#h_pool1 = max_pool_2x2(h_conv1)  #对数据进行池化，输出形状为14*14*32
#
##定义卷积层2
#W_conv2 = weight_variable([5,5,32,64])  #5*5是卷积核的大小，32是输入数据的通道，64是输出数据的通道
#b_conv2 = bias_variable([64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)  #输出形状为14*14*64
#h_pool2 = max_pool_2x2(h_conv2)  #对数据进行池化，输出形状为7*7*64
#
##全连接层1
#W_fc1 = weight_variable([7*7*64,1024])
#b_fc1 = bias_variable([1024])
#
###将数据展平，转换：[n_samples,7,7,64]----->[n_samples,7*7*64]
#h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#
##全连接层2
#W_fc2 = weight_variable([1024,10])  #输入时1024，输出是10（类别是热码）
#b_fc2 = bias_variable([10])
#prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
#
#
#
#
#
##使用交叉熵损失函数
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices = [1]))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
#
#sess = tf.Session()
#sess.run(tf.initialize_all_variables())
#
#
#
#for i in range(1000):
#    batch_xs,batch_ys = mnist.train.next_batch(100)  #分批次读取数据
#    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#    
#    #每训练50次，对测试数据进行测试以下
#    if i%50==0:
#        print(compute_accuracy(mnist.test.images,mnist.test.labels))
    
    
    
    
#######使用saver#################
import tensorflow as tf
import numpy as np

###保存的数据
#W = tf.Variable([[1,2,3],[4,5,6]],dtype = tf.float32,name = 'weights')  #2行3列
#b = tf.Variable([[1,2,3]],dtype = tf.float32,name = 'biases')    #1行3列
#
#init = tf.initialize_all_variables()
#saver = tf.train.Saver()  #声明一个saver
#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess,'my_net/mytext.ckpt')  #将数据保存到指定文件夹中


#restore variables：加载参数
####——————必须保证加载的数据类型和形状是一样的-----------
#W = tf.Variable(np.arange(6).reshape((2,3)),dtype = tf.float32,name = 'weights')#生成6个数据，改变其形状为2行3列
#b = tf.Variable(np.arange(3).reshape((1,3)),dtype = tf.float32,name = 'biases')
#
##不需要init()这一步进行初始化
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    saver.restore(sess,'my_net/mytext.ckpt')
#    print('weights:',sess.run(W))
#    print('biases:',sess.fun(b))
    

#######循环神经网络RNN#####LSTM是长短期记忆网络
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#
###加载数据
#mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
#
###定义参数
#lr = 0.001   #学习率
#training_iters = 100000   #学习迭代次数
#batch_size = 128    #批量大小
#display_step = 10  #每训练10步,展示一下结果
#
#n_inputs = 28  #即mnist数据输入的形状：28*28
#n_steps = 28  #time steps
#n_hidden_unis = 128   #隐含层神经元
#n_classes = 10   #类别是10个
#
##为数据分配输入空间
#x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
#y = tf.placeholder(tf.float32,[None,n_classes])
#
#
##定义权值和阈值
#Weights = {
#        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
#        'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
#        }
#biases = {  #值都是0.1
#        'in':tf.Variable(tf.constant(0.1,shape = [n_hidden_unis,])),
#        'out':tf.Variable(tf.constant(0.1,shape = [n_classes,])) 
#        }
#
#def RNN(X,weights,biases):
#   
#    ############输入数据
#    #X(128batch,28steps,28inputs)---->(128*28,28inputs)  将3维数据转换成二维数据
#    X = tf.reshape(X,[-1,n_inputs])
#    ###X_in ==([128batch*28steps,128hidden])
#    X_in = tf.matmul(X,Weights(['in'])) +biases('in')
#    #X_in --->(128batch,28steps,128hidden)
#    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])  #将数据从二维转换成三维
#    
#    
#    ##############中间的cell
#    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias = 1.0,state_is_tuple = True)
#    #lstm_cell划分成（c_state,m_state）------(主cell,子cell)
#    _init_state = lstm_cell.zero_state(batch_size,dtype = tf.float32)
#    #outputs是list,保存了每一阶段的输出结果，states是最后一个阶段的state，包含c_state和m_state
#    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state = _init_state,time_major = False)
#
#
#
#    ###############输出结果:此处的states[1]（m_state）指的就是outputs的最后一个结果
#    results = tf.matmul(states[1],weights['out']) + biases['in']
#    
#    return results
#    
#    
#pred = RNN(x,Weights,biases)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))  #使用交叉熵来计算损失
#train_op = tf.train.AdamOptimizer(lr).minimize(cost)  #降低损失率
#
#correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  #判断预测的结果和真值是否一致，是为1，否为0
#accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))  #对值进行转换，然后求均值
#
#init = tf.initialize_all_variables()
#
#
#with tf.Session() as sess:
#    sess.run(init)
#    step = 0
#    while step * batch_size < training_iters:
#        batch_xs,batch_ys = mnist.train.next_batch(batch_size)  #取下一批次的数据
#        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
#        sess.run([train_op],feed_dict ={
#                x:batch_xs,
#                y:batch_ys })
#    
#    #每训练20步就对数据进行一次预测
#        if step%20==0:
#            print(sess.run(accuracy,feed_dict ={
#                    x:batch_xs,
#                    y:batch_ys
#                    }))
#        step +=1




#########RNN lstm的例子
#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#
#BATCH_START = 0
#TIME_STEPS = 20
#BATCH_SIZE = 50
#INPUT_SIZE = 1
#OUTPUT_SIZE = 1
#CELL_SIZE = 10
#LR = 0.006   #学习率
#BATCH_START_TEST = 0
#
#
##得到数据
#def get_batch():
#    global BATCH_START,TIME_STEPS
#    #xs shape(50batch,20steps)
#    
#    xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape(10*BATCH_SIZE,TIME_STEPS)
#    seq = np.sin(xs)
#    res = np.cos(xs)
#    BATCH_START += TIME_STEPS
#    
#    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]
#
#
#class LSTMRNN(object):
#    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
#        self.n_steps = n_steps
#        self.input_size = input_size
#        self.output_size = output_size
#        self.cell_size = cell_size
#        self.batch_size = batch_size
#        with tf.name_scope('inputs'):
#            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name = 'xs')
#            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name = 'ys')
#        with tf.name_scope('in_hidden'):
#            self.add_input_layer()
#        with tf.variable_scope('LSTM_cell'):
#            self.add_cell()
#        with tf.variable_scope('out_hidden'):
#            self.add_output_layer()
#        with tf.name_scope('cost'):
#            self.compute_cost()
#        with tf.name_scope('train'):
#            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
#    
#    
#    def add_input_layer(self,):
#        #先把数据从3维转换成2维
#        l_in_x = tf.reshape(self.xs,[-1,self.input_size],name='2_2D')  #(batch*n_steps,in_size)
#        #Ws_in(in_size,cell_size)
#        Ws_in = self._weight_variable([self.input_size,self.cell_size])
#        #bs(cell_size,)
#        bs_in = self._bias_variable([self.cell_size,])
#        
#        #l_in_y = (batch*n_steps,cell_size)
#        with tf.name_scope('Wx_plus_b'):
#            l_in_y = tf.matmul(l_in_x,Ws_in) + bs_in
#            
#        
#        #将数据从2维转换成3维数据：reshape l_in_y---->(batch,n_steps,cell_size)
#        self.l_in_y =tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name = '2_3D')
#        
#  
#        
#    def add_cell(self):
#        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias = 1.0,state_is_tuple = True)
#        with tf.name_scope('initial_scope'):
#            #初始化状态的值都为0
#            self.cell_init_state = lstm_cell.zero_state(self.batch_size,dtype = tf.float32)
#        
#        #cell.output是一个list,保留了每一个输出的结果
#        self.cell_output,self.cell_final_state =tf.nn.dynamic_rnn(
#                lstm_cell,self.l_in_y,initial_state = self.cell_init_state,time_major = False
#                )
# 
#    
#    def add_output_layer(self):
#        #将3维数据转换成2维数据：shape----(batch*steps,cell_size)
#        l_out_x = tf.reshape(self.cell_outputs,[-1,self.cell_size],name = '2_2D')
#        Ws_out = self._weight_variable([self.cell_size,self.output_size])
#        bs_out = self._bias_variable([self.output_size,])
#        
#        #shape = (batch*steps,output_size)
#        with tf.name_scope('Wx_plus_b'):
#            self.pred = tf.matmul(l_out_x,Ws_out) + bs_out
#            
#
#    
#    def compute_cost(self):
#        #得到每一步的loss
#        losses = tf.nn.seq2seq.sequenee_loss_by_example(
#                [tf.reshape(self.pred,[-1],name = 'reshape_pred')],
#                [tf.reshape(self.ys,[-1],name='reshape_target')],
#                [tf.ones([self.batch_size*self.n_steps],dtype = tf.float32)],
#                average_across_timesteps = True,
#                softmax_loss_function = self.msr_error,
#                name = 'lossess'
#                )
#        #把所有的loss加起来
#        with tf.name_scope('average_cost'):
#            self.cost = tf.div(
#                    tf.reduce_sum(losses,name = 'losses_sum'),
#                    tf.cast(self.batch_size,tf.float32),
#                    name = 'average_cost'                    
#                    )
#            tf.scalar_summary('cost',self.cost)
#            
#            
#            
#            
#    def msr_error(self,y_pre,y_target):  #计算误差
#        return tf.square(tf.sub(y_pre,y_target))
#    
#    def _weight_variable(self,shape,name = 'weights'):
#        initializer = tf.random_normal_initializer(mean = 0.,stddev =1.)
#        return tf.get_variable(shape =shape,initializer = initializer,name = name)
#    
#    def _bias_variable(self,shape,name='biases'):
#        initializer = tf.constant_initializer(0.1)  #初始值都为0.1
#        return tf.get_variable(name = name,shape = shape,initializer = initializer)
#
#
#if __name__ == '__main__':
#    model = LSTMRNN(TIME_STEPS,INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,BATCH_SIZE)
#    sess = tf.Session()
#    merged = tf.summary.merge_all()
#    writer = tf.summary.FileWriter('logs',sess.graph)
#    
#    sess.run(tf.initialize_all_variables())
#    
#    
#    
#    ######可视化数据
#    plt.ion()  #连续绘制图
#    plt.show()
#    
#    for i in range(200):
#        seq,res,xs = get_batch()
#        if i ==0:
#            feed_dict = {
#                    model.xs:seq,
#                    model.ys:res,
#                    }
#            
#        else:
#            feed_dict = {
#                    model.xs:seq,
#                    model.ys:res,
#                    #使用上一个阶段结束的state作为这一个阶段开始的state
#                    model.cell_init_state:state  #use last state as the initial state for this run
#                    
#                    }
#         #计算模型的损失和对模型进行 反向训练  
#        _,cost,state,pred = sess.run(
#                [model.train_op,model.cost,model.cell_final_state,model.pred],
#                feed_dict = feed_dict             
#                )
#        
#        #可视化数据：以下变量都是二维数据
#        plt.plot(xs[0,:],res[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')
#        plt.ylim((-1.2,1.2))
#        plt.draw()
#        plt.pause(0.3)
#        
#        if i %20==0:  #每训练20次将参数保存到指点文件中一次
#            print('cost:',round(cost,4))
#            result = sess.run(merged,feed_dict)
#            writer.add_summary(result,i)
    

            











##############name_scope的用法#############
##from __future__ import print_function
#import tensorflow as tf
#tf.set_random_seed(1)   #repreducible
#
#
#with tf.name_scope("a_name_scope"):
#    #第一种方式创建变量
#    initializer = tf.constant_initializer(value = 1)  #定义一个常量值
#    var1 = tf.get_variable(name = 'var1',shape = [1],dtype = tf.float32,initializer = initializer)
#    #第二中方式创建变量
#    var2 = tf.Variable(name = "var2",initial_value = [2],dtype = tf.float32)
#    var21 = tf.Variable(name = "var2",initial_value = [2.1],dtype = tf.float32)
#    var22 = tf.Variable(name = "var2",initial_value =[2.2],dtype = tf.float32)
#    
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    
#    
#    ####可以发现使用第一种方式创建变量时，变量的名字前面没有之前的name_scope，
#    #而使用第二种方式则会让变量名前加上name_scope
#    print(var1.name)  #打印名字
#    print(sess.run(var1))   #打印值
#    print(var2.name)
#    print(sess.run(var2))
#    
#    print(var21.name)  #由于其名字与前面的名字一样，所以会自动修改其名字
#    print(sess.run(var21))




########variable_scope()的用法
#import tensorflow as tf
#tf.set_random_seed(1)  #reproducible
#
#with tf.variable_scope("a_variable_scope") as scope:
#    initializer = tf.constant_initializer(value = 3)
#    var3 = tf.get_variable(name = 'var3',shape = [1],dtype = tf.float32,initializer = initializer)
#    scope.reuse_variables()
#    var3_reuse = tf.get_variable(name = 'var3')  #其实其与var3是同一个变量
#    var4 = tf.Variable(name = 'var4',initial_value = [4],dtype = tf.float32)
#    var4_reuse = tf.Variable(name = 'var4',initial_value = [4],dtype = tf.float32)
#
#
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    print(var3.name)
#    print(sess.run(var3))
#    print(var3_reuse.name)
#    print(sess.run(var3_reuse))
#    
#    print(var4.name)
#    print(sess.run(var4))
#    print(var4_reuse)
#    print(sess.run(var4_reuse))
    

#######批量归一化的使用#####




























