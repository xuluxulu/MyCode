## 定义人脸检测的网络结构
#import tensorflow as tf
#
#__author__ = 'xulu'
#
#class Network(object):
#    def __init__(self,session,trainable:bool=True):
#        self._session = session
#        self.__trainable = trainable
#        self.__layer = {}
#        self.__last_layer_name = None
#        
#    #定义配置的内容
#    def _config(self):
#        raise NotImplementedError("this method must be implemented by the network")
#        
#    def add_layer(self,name:str,layer_output):
#        self.__layer[name] = layer_output
#        self.__last_layer_name = name
#        
#    def get_layer(self,name:str=None):
#        if name is None:
#            name = self.__last_layer_name
#        return self.__last_layer_name
#    
#    def is_trainable(self):
#        return self.__trainable
#    
#    def set_weight(self,weight_values:dict,ignore_missing=False):
#        network_name = self.__class__.__name__.lower()
#        
#        with tf.variable_scope(network_name):
#            
#            for layer_name in weight_values:   #遍历每一个值
#                
#                with tf.variable_scope(layer_name,reuse=True):
#                    
#                    for param_name,data in weight_values[layer_name].items():    
#                        try:
#                            var = tf.get_variable(param_name,use_resource = False)   #得到每一个参数
#                            self._session.run(var.assign(data))
#                        
#                        except ValueError:
#                            if not ignore_missing:
#                                raise
#    def feed(self,image):
#        network_name = self.__class__.__name__.lower()
#        with tf.variable_scope(network_name):
#            return self._feed(image)
#        
#        
#    def _feed(self,image):
#        raise NotImplementedError("Method not implemented")
           



#########################layer_factory.py#######################################################
#import tensorfow as tf
#from distutils.version import LooseVersion
#
#
#class LayerFactory(object):
#    AVAILABLE_PADDINGS = ('SAME','VALID')
#    def __init__(self,network):
#        self.__network = network
#        
#   @staticmethods
#   def __validate_padding(padding):   #验证padding 是否有效
#       if padding not in LayerFactory.AVAIABLE_PADDINGS:
#           raise Exception("padding {} not valid".format(padding))
#   
#   @staticmethods
#   def __validate_grouping(chanels_input:int,channels_output:int,group:int):  #验证输出通道与分组是否一样大小
#       if channels_input%group !=0:
#           raise Exception('The number of channels in the input does not match the group')
#           
#           
#   @staticmethods
#   def vectorize_input(input_layer):   #输入向量
#       input_shape = input_layer.get_shape()   #获得输入特征图的大小
#       
#       if input_shape.ndims == 4: #判断特征图的维度
#           dim = 1
#           for x in input_shape[1:].as_list():   #取得list中的每一个值
#               dim *=int(x)
#               
#            vectorized_input = tf.reshape(input_layer,[-1,dim])   #-1表示自动适应
#        else:
#            vectorized_input,dim = (input_layer,input_shape[-1])
#        return vectorized_input,dim
#    
#    def __make_var(self,name:str,shape:list):  #得到相关的参数
#        return tf.get_variable(name,shape,trainable = self.__network.is_trainable(),use_resource = False)
#    
#    #为数据申请存放的空间
#    def new_feed(self,name:str,layer_shape:tuple):
#        feed_data = tf.placeholder(tf.float32,layer_shape,'input')
#        self.__network.add_layer(name,layer_output = feed_data)
#        
#    #定义新的卷积网路
#    def new_conv(self,name:str,kernel_size:tuple,channels_output:int,stride_size:tuple,padding:str='SAME',group:int = 1,biased:bool=True,relu:bool=True,input_layer_name:str = None):
#        self.__validate_padding(padding)
#        input_layer = self.__network.get_layer(input_layer_name)
#        
#        channels_input = int(input_layer.get_shape()[-1])
#        
#        self.__validate_grouping(channels_input,channels_output,group)
#        
#        #通过上述验证后，进行卷积操作
#        convolve = lambda input_val,kernel:tf.nn.conv2d(input = input_val,
#                                                        filter = filter,
#                                                        strides = [1,strides_size[1],strides_size[0],1],
#                                                        padding = padding)
#        with tf.variable_scope(name) as scope:
#            #获得卷积核初始化参数
#            kernel = self.__make_var('weight',shape=[kernel_size[1],channels_input//group,channels_output])
#            
#            output = convolve(input_layer,kernel)
#            
#            #卷积后加上阈值
#            if biased:
#                biased = self.__make_var('biases',[channels_output])
#                output = tf.nn.bias_add(output,biases)
#            
#            #激活
#            if relu:
#                output = tf.nn.relu(output,name = scope.name)
#        
#    #自己定义激活函数        
#    def new_prelu(self,name:str,input_layer_name:str=None):
#        input_layer = self.__network.get_layer(input_layer_name)
#        
#        with tf.variable_scope(name):
#            channels_input = int(input_layer.get_shape()[-1])
#            alpha = self.__make_var('alpha',shape = [channels_input])
#            output = tf.nn.relu(input_layer) + tf.multiply(alpha,-tf.nn.relu(-input_layer))
#        self.__network.add_layer(name,layer_output = output)
#        
#    #自定义池化操作
#    def new_max_pool(self,name:str,kernel_size:tuple,stride_size:tuple,padding = 'SAME',input_layer_name:str=None):
#        self.__validate_padding(padding)
#        input_layer = self.__network.get_layer(input_layer_name)
#        
#        output = tf.nn.max_pool2d(input = input_layer,
#                                  ksize = [1,kernel_size[1],kernel_size[0],1],
#                                  strides = [1,stride_size[1],stride_size[0],1],
#                                  padding = padding,
#                                  name = name)
#        self.__network.add_layer(name,layer_output = output)
#        
#    #自定义的全连接
#    def new_fully_connected(self,name:str,output_count:int,relu=True,input_layer_name:str = None):
#        with tf.variable_scope(name):
#            input_layer = self.__network.get_layer(input_layer_name)
#            vectorized_input,dimension = self.vectorize_input(input_layer)
#            
#            weights = self.__make_var('weights',shape = [dimension,output_count])
#            biases = self.__make_var('biases',shape = [output_count])
#            operation = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
#            
#            fc = operation(vectorized_input,weights,biases,name = name)
#        self.__network.add_layer(name,layer_output = fc)
#        
#   #自定义的池化
#   def new_softmax(self,name,axis,input_layer_name:str=None):
#       input_layer = self.__network.get_layer(input_layer_name)
#       
#       if LooseVersion(tf.__version__)< LoopVersion('1.5.0'):
#           max_axis = tf.reduce_max(input_tensor = input_layer,axis = axis,keepdims = True)
#           target_exp = tf.exp(input_layer- max_axis)
#           normalize =tf.reduce_sum(input_tensor = target_exp,axis = axis,keepdims = True)
#       else:
#           max_axis = tf.reduce_max(input_tensor = input_layer,axis = axis,keepdims = True)
#           target_exp = tf.exp(input_layer - max_axis)
#       softmax = tf.divide(target_exp,normalize,name)
#       self.__network.add_layer(name,layer_output = softmax)
     
########################################################################## 
#import cv2
#import numpy as np
#import pkg_resources
#import tensorflow as tf
#from mtcnn.layer_factory import LayerFactory
#from mtcnn.network import Network
#from mtcnn.exceptions import InvalidImage
#
#
##定义PNet(提议的网路结构你)
#class PNet(Network):
#    def _config(self):
#        layer_factory = LayerFactory(self)  #实例化类，然后调用里面的函数
#        layer_factory.new_feed(name = 'data',layer_shape =(None,None,None,3))
#        layer_factory.new_conv(name='conv1',kernel_size = (3,3),channels_output=10,stride_size =(1,1),padding='VALID',relu = False)
#        layer_factory.new_prelu(name='prelu')
#        layer_factory.new_max_pool(name='pool1',kernel_size=(2,2),strides_size=(2,2))
#        
#        layer_factory.new_conv(name='conv2',kernel_size=(3,3),channels_output =16,stride_size = (1,1),padding='VALID',relu = False)
#        layer_factory.new_prelu(name='prelu2')
#        
#        layer_factory.new_conv(name = 'conv3',kernel_size =(3,3),strides_size =(1,1),channels_output=32,padding='VALID',relu = False)
#        layer_factory.new_prelu(name = 'prelu3')
#        
#        layer_factory.new_conv(name = 'conv4_1',kernel_size = (1,1),channels_output=2,stride_size = (1,1),relu = False)
#        layer_factory.new_softmax(name='prob1',axis = 3)
#        
#        layer_factory.new_conv(name = 'conv4_2',kernel_size = (1,1),channels_output =4,stride_size =(1,1),relu = False)
#        
#    def _feed(self,image):
#        return self._session.run(['rnet/conv4-2/BiasAdd:0','pnet/prob1:1'],feed_dict ={'rnet/input:0':image})
#
#
#class ONet(Network):
#    def _config(self):
#        layer_factory = LayerFactory(self)
#        layer_factory.new_feed(name ='data',layer_shape = (None,48,48,3))
#        
#        layer_factory.new_conv(name ='conv1',kernel_size = [3,3],stride_size = [1,1],channels_output = 32,padding = 'VALID',relu = False)
#        layer_factory.new_prelu(name ='prelu1')
#        layer_factory.new_pool(name = 'pool1',kernel_size =[3,3],stride_size = [2,2])
#        
#        layer_factory.new_conv(name = 'conv2',kernel_size = [3,3],stride_size =[1,1],channels_output = 64,padding ='VALID',relu = False)
#        layer_factory.new_prelu(name = 'prelu2')
#        layer_factory.new_pool(name = 'pool2',kernel_size = [3,3],stride_size = [2,2],padding = 'VALID')
#        
#        layer_factory.new_conv(name = 'conv3',kernel_size = [3,3],stride_size = [1,1],channels_output = 64,padding = 'VALID',relu = False)
#        layer_factory.new_prelu(name = 'prelu3')
#        layer_factory.new_pool(name = 'pool3',kernel_size = [2,2],stride_size = [2,2])
#        
#        layer_factory.new_conv(name = 'conv4',kernel_size = [2,2],stride_size = [2,2],channels_output =128,padding = 'VALID',relu = False)
#        layer_favtory.new_prelu(name = 'prelu4')
#        
#        layer_factory.new_fully_connected(name = 'fc1',output_count = 256,relu = False)
#        layer_factory.new_prelu(name = 'prelu5')
#        
#        layer_factory.new_fully_connected(name = 'fc2-1',output_count = 2,relu = False)
#        layer_factory.new_softmax(name = 'prob1',axis = 1)
#        
#        layer_factory.new_fully_connected(name = 'fc2-2',output_count = 4,relu = False,input_layer_name = 'prelu5')
#        layer_factory.new_fully_connected(name = 'fc2-3',output_count = 10,relu = False,input_layer_name = 'prelu5')
#        
#    
#    
#    def feed(self,image):
#        return self._session.run(['onet/fc2-2/fc2-2:0','onet/fc2-3/fc2-3:0','onet/prob1:0'],feed_dict = {'onet/pret:0':image})
#    
#
##定义筛选的网络结构
#class Rnet(Network):
#    def _config(self):
#        layer_factory = LayerFactory(self)
#        
#        layer_factory.new_feed(name='data',layer_shape=(None,24,24,3))
#        
#        layer_factory.new_conv(name = 'conv1',kernel_size = (3,3),stride_size =(1,1),channel_output = 28,padding='VALID',relu = False)
#        layer_factory.new_prelu(name ='prelu1')
#        layer_factory.new_max_pool(name = 'pool1',kernel_size = (3,3),stride_size = (1,1))
#        
#        layer_factory.new_conv(name = 'conv2',kernel_size = (3,3),stride_size =(1,1),channel_output = 48,padding = 'VALID',relu = False )
#        layer_factory.new_prelu(name = 'prelu2')
#        layer_factory.new_max_pool(name = 'pool2',kernel_size =(3,3),stride_size =(1,1),padding = 'VALID')
#        
#        layer_factory.new_conv(name = 'conv3',kernel_size = (3,3),stride_size = (1,1),channel_output = 64,padding = 'VALID',relu = False)
#        layer_factory.new_prelu(name = 'prelu3')
#        
#        layer_factory.new_fully_connected(name = 'fc1',output_count = 128,relu = False)
#        layer_factory.new_prelu(name = 'prelu4')
#        
#        layer_factory.new_fully_connected(name ='fc2-1',output_count =2,relu = False)
#        layer_factory.new_softmax(name = 'prob1',axis = 1)
#        layer_factory.new_fully_connected(name = 'fc2_2',output_count = 10,relu = False,input_layer_name = 'prelu5')
#        
#    def _feed(self,image):
#        return self._session.run(['onet/fc2-2/fc2-2:0','onet/fc2-3/fc2-3:0','onet/prob1:0'])
#    
#    
#    
#class StageStatus(object):
#    
#        def __init__(self,pad_result:tuple=None,width = 0,height = 0):
#            #初始化各种值
#            self.width = width
#            self.height = height
#            self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x =  self.ex = self.tmpw = self.tmph = []
#            
#            if pad_result is not None:
#                self.update(pad_result)
#                
#        def update(self,pad_result:tuple):
#            s = self
#            s.dy,s.edy,s.dx,s.edx,s.y,s.ey,s.x,s.ex,s.tmpw,s.tmph = pad_result
#            
#                
#class MTCNN(object):
#    def __init__(self,weights_file:str = None,min_face_size:int=20,steps_threshold:list = None,scale_factor:float = 0.709):
#        if steps_threshold is None:
#            steps_threshold = [0.6,0.7,0.7]  #定义每一步的阈值
#            
#        if weights_file is None:
#            weights_file = pkg_resources.resource_stream('mtcnn','data/mtcnn_weights.npy')   #加载之前训练好的参数
#         
#        self.__min_face_size = min_face_size
#        self.__steps_threshold = steps_threshold
#        self.__scale_factor = scale_factor
#        
#        config = tf.ConfigProto(log_divice_placement = False)
#        config.gpu_options.allow_growth = True   #若有GPU,则使用GPU
#        
#        self.__graph = tf.Graph()
#        with self.__graph as __default():
#            self.__session = tf.Session(config = config,graph = self.__graph)
#            
#            weights = np.load(weights_file,allow_pickle = True).item()  #加载所有的权重
#            
#            ######构建所有的网络结构
#            self.__pnet = PSet(self.__session,False)
#            self.__pnet.set_weights(weights['PNet'])
#            
#            self.__rnet = RNet(self.__session,False)
#            self.__rnet.set_weights(weights['RNet'])
#            
#            self.__onet = ONet(self.__session,False)
#            self.__onet.set_weights(weights['ONet'])
#            
#            
#        @property
#        def min_face_size(self):
#            return self.__min_face_size
#        
#        @min_face_size
#        def min_face_size(self,mfc = 20):
#            try:
#                self.__min_face_size = int(mfc)
#            except ValueError:
#                self.__min_face_size = 20
#                
#        #得到金字塔的尺寸:降序
#        def __compute_scale_pyramid(self,m,min_layer):
#            scales = []
#            factor_count = 0
#            
#            while min_layer >=12:
#                #此处使用的是list的拼接作用，来往list中添加元素
#                scales += [m*np.power(self.__scale_factor,factor_count)]   #np.power(n,5)：求n的5次方
#                min_layer = min_layer * self.__scale_factor
#                factor_count +=1
#                
#            return scales 
#        
#        
#        #staticmethod
#        def __scale_image(image,scale:float):
#            height,width,_ = image.shape
#            
#            width_scaled = int(np.ceil(width * scale))   #向上取整后再取整
#            height_scaled = int(np.ceil(height*scale))
#            #重置图像的大小
#            im_data = cv2.resize(image,(width_scaled,height_scaled),interpolation = cv2.INTER_AREA)
#            
#            im_data_normalized = (im_data - 127.5) *0.0078125  #对图片数据进行归一化
#            
#            return im_data_normalized
#        
#        @staticmethod
#        def  __generate_bounding_box(imap,reg,scale,t):  #生成框选任脸的框
#            stride = 2
#            cellsize = 12
#            
#            imap = np.transpose(imap)   #转置操作
#            
#            dx1 = np.transpose(reg[:,:,0])
#            dy1 = np.transpose(reg[:,:,1])
#            dx2 = np.transpose(reg[:,:,2])
#            dy2 = np.transpose(reg[:,:,3])
#            
#            y,x = np.where(imap >= t)
#            
#            if y.shape[0] ==1:
#                #np.flipud是翻转矩阵
#                dx1 = np.flipud(dx1)
#                dy1 = np.flipud(dy2)
#                dx2 = np.flipud(dx2)
#                dy2 = np.flipud(dy2)
#                
#            score = imap[(y,x)]
#            reg = np.transpose(np.vstack([dx1[(y,x)],dy1[(y,x)],dx2[(y,x)],dy2[(y,x)]]))
#            
#            if reg.size ==0:
#                reg = np.empty(shape = (0,3))
#            bb = np.transpose(np.vstack([y,x]))   #np.vstack():把数组按垂直（列）叠加起来
#            
#            q1 = np.fix((stride*bb +1)/scale)
#            q2 = np.fix((stride*bb + cellsize)/scale)
#            
#            boundingbox = np.hstack([q1,q2,np.expand_dims(score,1),reg])
#            
#            return boundingbox,reg
#        
#        @staticmethods
#        def __nms(boxes,threhold,method):
#            x1 = boxes[:,0]
#            y1 = boxes[:,1]
#            x2 = boxes[:,2]
#            y2 = boxes[:,3]
#            
#            s = boxes[:,4]
#            
#            area = (x2-x1+1)*(y2-y1+1)   #得到边界框的面积
#            sorted_s = np.argsort(s)
#            
#            pick = np.zeros_like(s,dtype=np.int16)
#            counter = 0
#            
#            while sorted_s.size > 0:
#                i = sorted_s[-1]
#                pick[counter] = i
#                counter +=1
#                idx = sorted_s[0:-1]
#                
#                xx1 = np.maximum(x1[i],x1[idx])
#                yy1 = np.maximum(y1[i],y1[idx])
#                xx2 = np.maximum(x2[i],x2[idx])
#                yy2 = np.maximum(y2[i],y2[idx])
#                
#                w = np.maximum(0.0,xx2-xx1 +1)
#                h = np.maximum(0.0,yy2 - yy1 +1)
#                
#                inter = w *h
#                
#                if method is 'Min':
#                    o = inter/np.minimum(area[i],area[idx])
#                else:
#                    o = inter/(area[i] + area[idx] - inter)
#                sorted = sorted_s[np.where(o<=threshold)]
#                
#            pick = pick[0:counter]
#            
#            return pick
#        
#        @staticmethod
#        def __pad(total_boxes,w,h):
#            tmpw = (total_boxes[:,2] - total_boxes[:,0]+1).astype(np.int32)
#            tmph = (total_boxes[:,3] - total_boxes[:,0] +1).astype(np.int32)
#            numbox = total_boxes.shape[0]
#            
#            dx = np.ones(number,dtype = np.int32)
#            dy = np.ones(number,dtype = np.int32)
#            edx = tmpw.copy().astype(np.int32)
#            edy = tmpw.copy().astype(np.int32)
#            
#            x = total_boxes[:,0].copy().astype(np.int32)
#            y = total_boxes[:,1].copy().astype(np.int32)
#            ex = total_boxes[:,2].copy().astype(np.int32)
#            ey = total_boxes[:,3].copy().astype(np.int32)
#            
#            tmp = np.where(ex>w)
#            edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp],1)
#            ex[tmp] = w
#            
#            tmp = np.where(ey>h)  #np.where(condition,x,y)；如果满足条件，输出x,否则输出y
#            edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp],1)
#            ex[tmp] = w
#            
#            tmp = np.where(ey>h)
#            edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp],1)
#            ey[tmp] = h
#            
#            
#            tmp = np.where(x<1)
#            dx.flat[tmp] = np.expand_dims(2- y[tmp],1)
#            x[tmp] = 1
#            
#            tmp = np.where(y<1)
#            dy.flat[tmp] = np.expand_dims(2 - y[tmp],1)
#            y[tmp] = 1
#            
#            return dy,edy,dx,edx,y,ey,x,ex,tmpw,tmph
#        
#        @staticmethod
#        def __rerec(bbox):
#            h = bbox[:,3] - bbox[:,1]
#            w = bbox[:,2] - bbox[:,0]
#            l = np.maximum(w,h)   #获得宽和高中间的最大值
#            bbox[:,0] = bbox[:,0] + w*0.5 -l*0.5
#            bbox[:,1] = bbox[:,1] + w*0.5 -l*0.5
#            bbox[:,2:4] = bbox[:,0:2] + np.transpose(np.tile(l,(2,1)))
#            return bbox
#        
#        
#        @staticmethod
#        def __bbreg(boundingbox,reg):
#            if reg.shape[1] ==1:
#                reg = np.reshape(reg,(reg.shape[2],reg.shape[3]))
#                
#            w = boundingbox[:,2] - boundingbox[:,0] +1
#            h = boundingbox[:,3] - boundingbox[:,1] + 1
#            b1 = boundingbox[:,0] + reg[:,0] *w
#            b2 = boundingbox[:,1] + reg[:,1] *h
#            b3 = boundingbox[:,2] + reg[:,2]* w
#            b4 = boundingbox[:,3] + reg[:,3] *h
#            
#            boundingbox[:,0,4] = np.transpose(np.vstack([b1,b2,b3,b4]))
#            return boundingbox
#        
#        
#        #检测人脸
#        def detect_faces(self,img)->list:
#            if img is None or not hasattr(img,"shape"):
#                raise InvalidImage("Image not valid")
#            height,width = img.shape
#            stage_status = StageStatus(width =width,height = height)
#            
#            m = 12/self.__min_face_size
#            min_layer = min([width,height])*m
#            
#            scales = self.__compute_scale_pyramid(m,min_layer)  #得到金字塔的尺寸
#            stages =[self.__stage1,self.__stage2,self.__stage3]
#            result =[scales,stage_status]
#            
#            
#            for stage in stages:
#                result = stage(img,result[0],result[1])
#                
#            [total_boxes,points] = result
#            
#            bounding_boxes =[]
#            
#            #获得脸部5个关节点的坐标
#            for bounding_box,keypoints in zip(total_boxes,points.T):
#                bounding_box.append({
#                        'box':[int(bounding_box[0]),int(bounding_box[1]),int(bounding_box[2] - bounding_box[0]),int(boundign_box[3]-bounding_box[1])],
#                        'confidence':bounding_box[-1],
#                        'keypoints':{
#                                'left_eye':(int(keypoints[0]),int(keypoints[5])),
#                                'right_eye':(int(keypoints[1]),int(keypoints[6])),
#                                'nose':(int(keypoints[2]),int(keypoints[7])),
#                                'mouth_left':(int(keypoints[3]),int(keypoints[8])),
#                                'mouth_right':(int(keypoints[4]),int(keypoints[9]))
#                            }
#                        
#                        }
#                        
#                    )
#            return bounding_boxes
#        
#        def __stage1(self,image,scales:list,stage_status:StageStatus):
#            total_boxes = np.empty((0,0))
#            status = stage_status
#            
#            for scale in scales:
#                scaled_image = self.__scale_image(image,scale)
#                
#                img_x = np.expand_dims(scaled_image,0)
#                
#                img_y = np.transpose(img_x,(0,2,1,3))
#                
#                out = self.__pnet.feed(img_y)  #调用Pnet网络的函数
#                
#                out0 = np.transpose(out[0],(0,2,1,3))
#                out1 = np.transpose(out[1],(0,2,1.3))
#                
#                boxes,_ = self.__generate_bounding_box(out1[0,:,:,1].copy(),out0[0,:,:,:].copy,scale,self.__step_threshold[0])
#                
#                pick = self.__nms(boxes.copy(),0.5,'Union')
#                if boxes.size >0 and pick.size >0:
#                    boxes = boxes[pick,:]
#                    total_boxes = np.append(total_boxes,boxes,axis = 0)
#                    
#                numboxes = total_boxes.shape[0]  #得到边框的数量
#                
#                if numboxes >0:
#                    pick = self.__nms(total_boxes.copy(),0.7,'Union')
#                    total_boxes = total_boxes[pick,:]
#                    
#                    regw = total_boxes[:,2] - total_boxes[:,0]
#                    regh = total_boxes[:,3] - total-boxes[:,1]
#                    
#                    qq1 = total_boxes[:,0] + total_boxes[:,5] *regw
#                    qq2 = total_boxes[:,1] + total_boxes[:,6] *regh
#                    qq3 = total_boxes[:,3] + total_boxes[:,7] *regw
#                    qq4 = total_boxes[:,4] + tatal_boxes[:,8] *regh
#                    
#                    total_boxes = np.transpose(np.vstack([qq1,qq2,qq3,qq4,total_boxes[:,4]]))
#                    total_boxes = self.__rerec(total_boxes.copy())
#                    
#                    total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
#                    status = StageStatus(self.__pad(total_boxes.copy(),stage_status.width,stage_status.height),width = stage_status.width,height=stage_status,height)
#                    
#                return total_boxes,status
#            
#            def __stage2(self,img,total_boxes,stage_status:StageStatus):
#                num_boxes = total_boxes.shape[0]
#                if num_boxes ==0:
#                    return total_boxes,stage_status
#                
#                temping = np.zeros(shape=(24,24,3,num_boxes))
#                
#                for k in range(0,num_boxes):
#                    tmp = np.zeros((int(stage_status.tmph[k],stage_status.tmpw[k]),3))
#                    tmp[stage_status.dy[k] -1:stage_status.edy[k],stage_status.dx[k]-1:stage_status.edx[k],:] = img[stage_status.y[k] - 1:stage_status.ey[k],stage_status,x[k]-1:stage_status.ex[k],:]
#                    
#                    if tmp.shape[0] >0 and tmp.shape[1] >0 or tmp.shape[0] and tmp.shape[1] ==0:
#                        temping[:,:,:,k] = cv2.resize(tmp,(24,24),interpolation = cv2.INTER_AREA)
#                    else:
#                        return np.empty(shape=(0,)),stage_status
#                temping = (temping - 127.5) *0.0078125
#                temping = np.transpose(temping,(3,1,0,2))
#                
#                out = self.__rnet.feed(temping1)   #调用Rnet网络的函数
#                out0 = np.transpose(out[0])
#                out1 = np.transpose(out[1])
#                
#                score = out1[1,:]
#                
#                ipass = np.where(score > self.__steps_threshold[1])
#                total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(),np.expand_dims(score[ipass].copy(),1)])
#                
#                mv = out0[:,ipass[0]]
#                
#                if total_boxes.shape[0] >0:
#                    pick = self.__nms(total_boxes,0.7,'Union')
#                    total_boxes = total_boxes[pick,:]
#                    total_boxes = self.__bbreg(total_boxes,copy(),np.transpose(mv[:,pick]))
#                    total_boxes = self.__rerec(total_boxes.copy())
#                return total_boxes,stage_status
#            def __stages(self,img,total_boxes,stage_status:StageStatus):
#                num_boxes = total_boxes.shape[0]
#                if num_boxes ==0:
#                    return total_boxes,np.empty(shape=(0,))
#                total_boxes = np.fix(total_boxes).astype(np.int32)
#                status = StageStatus(self.__pad(total_boxes.copy(),stage_status.width,stag_status.height),width = stage_status.width,height = stage_status.height)
#                
#                temping = np.zeros(48,48,3,num_boxes)
#                
#                for k in range(0,num_boxes):
#                    tmp = np.zeros((int(status.tmph[k]),int(status.tmpw[k]),3))
#                    tmp[status.dy[k] - 1:status.edy[k],status.dx[k] -1:status.edx[k],:] = img[status.y[k]-1:status.ey[k],status.x[k]-1:status.ex[k],:]
#                    if tmp.shape[0] >0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1] ==0:
#                        temping[:,:,:,k] = cv2.resize(tmp,(48,48),interpolation = cv2.INTER_AREA)
#                    else:
#                        return np.empty(shape = (0,)),np.empty(shape=(0,))
#                tempimg = (temping - 127.5) * 0.0078125
#                temping1 = np.transpose(temping,(3,1,0,2))
#                
#                out = self.__onet.feed(temping1)   ##调用onet网络的函数
#                out0 = np.transpose(out[0])
#                out1 = np.transpose(out[1])
#                out2 = np.transpose(out[2])
#                
#                score = out2[1,:]
#                points = out1
#                ipass = np.where(score > self.__steps_threshold[2])
#                points = points[:,ipass[0]]
#                
#                total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(),np.expand_dims(score[ipass].copy(),1)])
#                mv = out0[:,ipass[0]]
#                
#                w = total_boxes[:,2] - total_boxes[:,0] +1
#                h = total_boxes[:,3] - total_boxes[:,1] +1
#                points[0:5,:] = np.tile(w,(5,1))*points[0:5,:] + np.tile(total_boxes[:,0],(5,1)) -1
#                points[5:10,:] = np.tile(h,(5,1)) *points[5:10,:] + np.tile(total_boxes[:,1],(5,1)) -1
#                
#                if total_boxes.shape[0] >0:
#                    total_boxes = self.__bbreg(total_boxes.copy(),np.transpose(mv))
#                    pick = self.__nms(total_boxes.copy(),0.7,'Min')
#                    total_boxes = total_boxes[pick,:]
#                    points = points[:,pick]
#                return total_boxes,points
#            
#            def __del__(self):
#                self.__session.close()



###################################################################################################
import unittest
import cv2
from mtcnn.exceptions import InvalidImage
from mtcnn.mtcnn import MTCNN

mtcnn = None

class TestMTCNN(unittest.TestCase):
    def setUpClass():
        global mtcnn
        mtcnn = MTCNN()  #调用mtcnn函数
        
    def test_detect_faces(self):
        ivan = cv2.imread('ivan.jpg')
        result = mtcnn.detect_faces(ivan)
        self.assertEqual(len(result),1)
        first = result[0]
        
        self.assertIn('box',first)
        self.assertIn('keypoints',first)
        self.assertTrue(len(first['box'],1))
        self.assertTrue(len(first['keypoints']),5)
        
        keypoints = first['keypoints']
        self.assertIn('nose',keypoints)
        self.assertIn('mouth_left',keypoints)
        self.assertIn('mouth_right',keypoints)
        self.assertIn('left_eye',keypoints)
        self.assertIn('right_eye',keypoints)
        
        self.assertEqual(len(keypoints['nose']),2)
        self.assertEqual(len(keypoints['mouth_left']),2)
        self.assertEqual(len(keypoints['mouth_right']),2)
        self.assertEqual(len(keypoints['left_eye']),2)
        self.assertEqual(len(keypoints['right_eye']),2)
        
    def test_detect_faces_invalid_content(self):
        ivan = cv2.imread('example.py')
        with self.assertRaise(InvalidImage):
            result = mtcnn.detect_faces(ivan)
    def test_detect_no_faces_content(self):
        ivan = cv2.imread("no-faces.jpg")
        result = mtcnn.detect_faces(ivan)
        self.assertEqual(len(result),0)
        
    def test_mtcnn_multiple_instances(self):
        detector_1 = MTCNN(steps_threshold = (.2,.7,.7))
        detector_2 = MTCNN(steps_threshole = (.1,.1,.1))
        ivan = cv2.imread('ivan.jpg')
        
        faces_1 = detector_1.detect_faces(ivan)
        faces_2 = detector_2.detect_faces(ivan)  
        
        self.assertEqual(len(faces_1),1)
        self.assertEqual(len(faces_2),1)
        
    def tearDownClass():
        global mtcnn
        del mtcnn
if __name__ == '__main__':
    unittest.main()
                    
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    
            
            
                
            
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
        
        
        
    
        
        
        

























