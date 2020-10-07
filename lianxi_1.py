#####int()  str()的用法
a = '12'
print(type(a))
print(a)
a = int(a)
print(type(a))
print(a)


#####一起申明变量，一起输出变量
#a1,a2,a3=1,2,3
#print(a1,a2,a3)


##while/for的用法
#condition = 1
#while condition<10:
#    print(condition)
#    condition = condition + 1

#example_list = [1,5,2,6,35,78,24,63,666,22]
#for i in example_list:
#    print(i)

#rang(10)  表示从0开始到9，步长为1
#for i in range(1,10,3):   #起始值为1，终止值为9，步长为3
#    print(i)



######定义函数的使用
#def sum(a,b):
#    c = a+b
#    return c
##w = sum(1,2)  #调用函数
##w = sum(b=5,a=2)
#w = sum(a=2)   #出错
#print(w)

######定义函数的默认参数
#def scale_car(price,colour='red',brand='aaa',is_second_hand=True):   #在参数给默认值时，只要有一个参数给给了默认值，该参数后的其他参数都要给默认值，否则在调用时会出错
#    print('price:',price,
#          'colour:',colour,
#          'brand:',brand,
#          'is_second_hand:',is_second_hand)
#    
#scale_car(1000,'block')   #返回的结果，‘block’将替代‘red’这个默认值



######定义全局变量和局部变量
#b = 10
#def func():
#    global b  #函数外部的参数想要通过无参的函数改变其值，必须通过在函数内部是使用‘global’对其进行再一次的申明
#    a = 20
#    b = b + 10
#    return a+100,b
#
#print(func())

#######文件的读/写
#写文件
#text = 'This is my first test \n This is last line'   #\n表示换行
#print(text)
##my_file = open('myfile.txt','w')   #myfile.txt这个文件若存在，则打开，若不存在则进行创建,'W'表示能对这个文件进行编写，‘r’表示只能对这个文件进行读
#my_file = open('myfile.txt','a')   #往文件里添加追加东西
#my_file.write(text)   #将定义的文档写入文件中
#my_file.close()   #关闭文件

#读文件
#file = open('myfile.txt','r')  #读取这个文件中的内容
##content = file.read()  #读取文件的全部内容
#
##first_content = file.readline()  #只读取第一行
##second_content = file.readline()  #读取文件的第二行内容
#content = file.readlines()  #把读取的内容放到list中
#print(content)
#file.close()


######定义一个类class:类名要大写
#class Calculator:
#    name = "good calculator"
#    price = 18
#    
#    def __init__(self,name,price,hight,width,weight):
#        self.name = name
#        self.price = price
#        self.hight = hight
#        self.width = width
#        self.weight = weight
#        self.add(1,5)   #调用自己的函数
#    
#    
#    
#    def add(self,x,y):
#        #print(self.name)
#        print(x+y)
#    def minus(self,x,y):
#        return x-y
#
#    def times(self,x,y):
#        return x*y
#    def divide(self,x,y):
#        return x/y
#cal = Calculator('bad calcultor',12,30,15,10)  #传入的参数必须和__init__()中的参数的数量一样
##print(cal.name)
#sum = cal.add(1,2)
#print(sum)


######input()的用法
#text = input('请输入一个number：')   #返回的是字符串
#text = int(text)
#if text ==1:
#    print('this is a good lucky')
#elif text ==2:
#    print('this is a medium')
#else:
#    print('this is so bad')
   


######tuple() list
#a_tuple = (45,25,3,1,85,11)
#anthor_tuple = 1,5,3,47,82,66 
#print(type(anthor_tuple))
#for i in a_tuple:
#    print(i)
#for i in anthor_tuple:
#    print(i)
#    
#
#
#a_list = [12,56,3,29,41]
#for i in a_list:
#    print(i)

#list的操作
#a = [21,3,5,6,22,33,1,5,0,6,8,2,7]
#a.append(34)  #在最后添加一个元素
#print(a)
#a.insert(1,0)   #在索引为1处添加一个0
#print(a)
#a.remove(0)  #去除a中第一个出现为0的元素
#print(a)
#print(a[1:5])   #输出索引为1/2/3/4的元素（即第二/三/四/五的元素）
#print(a.count(5))  #找到列表a中5出现的次数
#a.sort()  #从下到大排序
#print(a)
#a.sort(reverse=True)  #从大到小排序
#print(a)

######多维列表
#a = [1,2,3,4,5,6]
#multi_dim_a =[[1,2,3],
#              [2,3,4],
#              [3,4,5]]
#print(a[1])
#print(multi_dim_a[0][1])  #输出行的索引为0，列的索引为1的值


######字典（dict）key:value
#d1 = {'apple':1,'pear':2,'orange':3,'cc':[1,2,3,6],'ce':{'11':1}}
#del d1['ce']
#print(d1)
#d1['ce'] = {'11':1}
#print(d1)
#print(d1['cc'][1])
#print(d1['apple'])


#####导入时间模块
#import time
#print(time.localtime())  #获得当前本机的时间
#print(time.time())  
#
#from time import localtime
#from time import time
#print(localtime())
#print(time())

#from lianxi_2 import add
#add(1,2)  #直接调用导入的函数

#import lianxi_2
#lianxi_2.add(1,2)




######break/continue
#flag = True
#aa = [1,55,2,366,999,42,63,65,636,5]
#while flag:
#    for i in aa:
#        if i ==2:
#            #continue
#            break
#        print(i)
#    flag = False
#print('已经结束当前工作')




######try.........except:......else:.........
#try:
#    file = open('eeeee','r+')   #打开eeeee这个文件，但是这个文件不存在，所以会出现错误
#except Exception as e:
#    print(e)   #打印出错误
#    response = input('do you want to create a new file:')
#    if response =='y':
#        file = open('eeeee','w')
#    else:
#        pass
#else:
#    file.write('sssss')  #若文件存在，则往文件中写入‘sssss’
#file.close()
#



######zip
#a = [1,2,3]
#b = [5,6,7]
#c = list(zip(a,b))
#print(c)
#cc = list(zip(a,a,b))
#print(cc)
#
#for i,j in zip(a,b):
#    print(i,j)

####lambda的使用
#fun2 = lambda x,y:x+y
#a = fun2(2,3)
#print(a)



####map的使用
#fun2 = lambda x,y:x+y
#print(list(map(fun2,[2,3],[4,5])))  #x和y,这两个变量传入两个参数



#pickle是用来保存运行的结果供之后使用
#import pickle
#a_dict = {'da':111,2:[25,62],'23':{1:2,'33':1234}}
#file = open('aaa.pickle','wb')  #以二进制的形式往里面写入内容
#pickle.dump(a_dict,file)   #将内容写入文件中
#file.close()


#file = open('aaa.pickle','rb')   #读取二进制文件
#a_dict1 = pickle.load(file)
#file.close()
#print(a_dict1)


#with open('aaa.pickle','rb') as file:
#    a_dict = pickle.load(file)
#    print(a_dict)



####set的使用
#char_list = ['a','b','c','c','a']
#aa = set(char_list)
#print(aa)
#aa.add('x')
#print(aa)
#aa.remove('x')
#print(aa)
#aa.clear()
#print(aa)

    
#####多进程：multiprocessing
#import multiprocessing as mp
#def job(a,d):
#    print('ssssss')
#if __name__ == '__main__':
#    p1 = mp.Process(target = job,args = (1,3))   #定义一个进程：job是调用的函数名称，args是给job函数传的参数
#    p1.start()  #启动线程
#    p1.join()
#    print('wwww')
    


#import multiprocessing as mp
##把计算的多进程的结果放入队列中，供以后使用
#def job(q):
#    res = 0
#    for i in range(10):
#        res += i + i**2+ i**3
#    q.put(res)  #把计算的结果放入队列中
#    
#if __name__ == '__main__':
#    q = mp.Queue()   #声明一个队列
#    p1 = mp.Process(target = job,args = (q,))  #调用一个函数，并将申明的队列传入进去
#    p2 = mp.Process(target = job,args = (q,))   #args中传的参数中的‘，’表示之后可能还会传入其他参数
#    
#    #启动进程
#    p1.start()
#    p2.start()
#    p1.join()
#    p2.join()
#    #把放入队列中的两个值取出
#    res1 = q.get()  
#    res2 = q.get()
#    
#    print(res1+res2)
    
    
#######对比实验：多进程、多线程、只有一个主线程-----哪个运行速度快？结果表明：多进程最快（同一时刻有几个进程并行在运行），只有一个主线程居中，多线程最慢（它其实每一时刻还是只有一个线程在运行）
#import multiprocessing as mp
#import threading as td
#import time
#def job(q):
#    res = 0
#    for i in range(10000000):
#        res += i + i**2 + i**3
#    q.put(res)    #q是队列
#
##使用多进程的方式
#def multcore():
#    q = mp.Queue()  #申明一个队列
#    p1 = mp.Process(target = job,args = (q,))  #申明线程
#    p2 = mp.Process(target = job,args = (q,))
#    p1.start()
#    p2.start()
#    p1.join()
#    p2.join()
#    
#    res1 = q.get()
#    res2 = q.get()
#    print('process的结果:',res1 + res2)
#    
#
##只有一个主线程的方法
#def normal():
#    res = 0
#    for _ in range(2):
#        for i in range(10000000):
#            res += i + i**2 + i**3
#        print('normal:',res)
#    
#
#
##使用多线程的方法    
#def multithread():
#    q = mp.Queue()
#    t1 = td.Thread(target = job,args=(q,))  #申明进程
#    t2 = td.Thread(target = job,args=(q,))
#    t1.start()
#    t2.start()
#    t1.join()
#    t2.join()
#    #取得放入队列中的结果
#    res1 = q.get()
#    res2 = q.get()
#    print('在多进程中的结果：',res2+res1)
#    
#if __name__ =='__main__':
#    st = time.time()
#    normal()
#    st1 = time.time()
#    print('normal time:',st1 - st)
#    
#    st = time.time()
#    multithread()
#    st2 = time.time()
#    print('multithread time:',st2-st)
#    
#    
#    st = time.time()
#    multcore()
#    st3 = time.time()    
#    print('multcore time:',st3 -st)



######进程池的使用
#import multiprocessing as mp
#
#def job(x):
#    return x*x
#
#def multicore():
#    pool = mp.Pool()  #定义一个进程池:启动cpu的所有核   pool = mp.Pool(processes = 2)表示启用cpu的两个核
#    res = pool.map(job,range(10))  #往进程池里放调用的函数和参数，自动把参数分配给进程
#    print(res)   #直接就可以得到结果
#    
#    res = pool.apply_async(job,(2,))  #传入一个参数为2，得到一个结果
#    print(res.get())
#    multi_res = [pool.apply_async(job,(i,)) for i in range(10)]  #传入多个参数
#    print([res.get() for res in multi_res])  #从缓冲池中依次取出结果，放入到list中
#    
#if __name__ == '__main__':
#    multicore()


##使用锁共享内存
#import multiprocessing as mp
#import time
#def job(v,num,l):
#    l.Lacquire()  #上锁
#    for _ in range(10):
#        time.sleep(0.1)   #每一个暂停0.1秒
#        v.value += num   
#        print(v.value)
#    l.release()   #释放锁
#def multicore():
#    l = mp.Lock()  #申明一个锁
#    v = mp.Value('i',1)
#    p1 = mp.Process(target=job,args=(v,1,l))
#    p2 = mp.Process(target=job,args=(v,3,l))
#    
#    p1.start()
#    p2.start()
#    p1.join()
#    p2.join()
#    
#
#if __name__ == '__main__':
#    multicore()



####多进程
#import threading 
#import time
#
#def thread_job():
#    print('this is an added Thread,number is %s'%threading.current_thread())
#    print('T1 start\n')
#    for i in range(10):
#        time.sleep(0.1)  #暂停0.1秒
#    print('T1 finish\n')
#    
#def t2_job():
#    print('T2 start\n')
#    print('T2 finish\n')
#
#
#def main():
#    added_thread = threading.Thread(target = thread_job,name = 't1')   #定义一个线程，并给它一个函数，表示让它去做的工作
#    thread2 = threading.Thread(target = t2_job,name = 't2')
#    
#    added_thread.start()  #启动线程，线程开始运行
#    thread2.start()
#    
#    added_thread.join()
#    thread2.join()
#    
#    print('all done\n')  #所有线程执行完后，在执行主函数的相关操作
#    
#    
#if __name__ =='__main__':
#    main()


#####使用队列保存进程的结果
#import threading
#import time
#from queue import Queue
#
#def job(l,q):  #定义一个线程执行的任务
#    for i in range(len(l)):
#        l[i] = l[i]**2
#    q.put(l)   #把计算的结果放入到队列中
#
#def multithreading():
#    q = Queue()   #申明一个队列
#    threads = []
#    data = [[1,2,3],
#            [3,4,5],
#            [5,6,7],
#            [5,6,4]
#            ]
#    for i in range(4):
#        t = threading.Thread(target = job,args =(data[i],q))     #为每一个子list生成一个线程
#        t.start()  #启动这个子线程
#        threads.append(t)  #把每一个子线程加入到所定义的列表中
#    for thread in threads:
#        thread.join()
#        
#    #等到上述线程全部运行完了，接下来对结果进行操作
#    results =[]    #取出队列中的结果
#    for _ in range(4):
#        results.append(q.get())  #取出队列中的每一个结果
#    print(results)
#
#if __name__ == '__main__':
#    multithreading()

   
  
    

####使用多线程和只使用一个主线程的运行效率进行对比：多线程在每一时刻只能让一个进行运行，其需要进行多次读写，而一个主线程只需进行一次读写操作
#import threading
#from queue import Queue
#import copy
#import time
#
#def job(l,q):
#    res = sum(l)  #求和
#    q.put(res)

##多线程的方法
#def multithreading(l):
#    q = Queue()
#    threads = []
#    for i in range(4):
#        t = threading.Thread(target = job,args =(copy.copy(l),q),name = 'T%i'%i)
#        t.start()
#        threads.append(t)
#    [t.join() for t in threads]
#    
#    
#    total = 0
#    for _ in range(4):
#        total += q.get()  #把每一个线程求得的结果加起来
#    print(total)
#    
##只有一个主线程的方法   
#def normal(l):
#    total = sum(l)
#    print(total)
#    
#if __name__ == '__main__':
#    l = list(range(100))   #[0,1,2,....99]
#    print(l)
#    s_t = time.time()
#    normal(l*4)
#    print('normal:',time.time() - s_t)
#    
#    
#    s_t = time.time()
#    multithreading(l)
#    print('multithreading:',time.time() - s_t)

    
#####lock(锁)：使一个线程完整运行，让其结果供下一个线程使用或者下一个线程再运行：即二者相互不影响
#import threading
#
#def job1():
#    global lock
#    global A
#    lock.acquire()   #获得锁
#    for i in range(10):
#        A += 1
#        print('job1',A)
#    lock.release()   #释放锁
#    
#def job2():
#    global lock
#    global A
#    lock.acquire()
#    for i in range(10):
#        A +=10
#        print('job2',A)
#    lock.release()
#    
#if __name__ == '__main__':
#    lock = threading.Lock()  #申明一个锁
#    A = 0   #定义一个全局变量
#    t1 = threading.Thread(target = job1)
#    t2 = threading.Thread(target = job2)
#    t1.start()
#    t2.start()
#    t1.join()
#    t2.join()





######tkinter(图形化界面设计):label  button
#import tkinter as tk
#
#window = tk.Tk()  #声明一个窗口
#window.title('my window')   #窗口的名字
#window.geometry('200x100')  #窗口的大小：长x高
#
#var = tk.StringVar()  #tk中特定的字符串变量
#
##在窗口中添加一个标签（label）
##textvarible表示标签上显示的文字，bg是背景颜色，font表示字体的大小和字体类型，height表示label的高度，即字体的2倍高
#l = tk.Label(window,textvariable = var,bg ='green',font=('Arial',12),width = 15,height=2)
#l.pack() #标签放置的位置
#
#on_hit = False
#def hit_me():
#    global on_hit
#    if on_hit ==False:
#        on_hit = True
#        var.set('you hit me')  #给字符串变量赋值
#    else:
#        on_hit = False
#        var.set('')  #将该变量设置为空
#
##button
##通过这个按钮为label设置显示的参数
#b = tk.Button(window,text = 'hit me',width = 15,height = 2,command = hit_me)  #command是给这个按钮一个函数
#b.pack()  #按钮放置的位置
#
#window.mainloop()




######tkinter：Entry&text输入、文本框
#import tkinter as tk
#window = tk.Tk()  #申明一个窗口
#window.title('my window')   #给窗口命名
#window.geometry('200x200')  #窗口的大小：长*高 
#var = tk.StringVar()  #tk中特定的字符串变量   
#
#e =tk.Entry(window,show = '*')  #设定一个Entry输入框：输入的内容都以“*”显示
#e.pack()  #按钮放置的位置
#
#def insert_point():
#    var = e.get()   #获得输入文本框中的内容
#    t.insert('insert',var)   #把内容插入到文本框中指针指的位置
#
#def insert_end():
#    var = e.get()
#    t.insert('end',var)  #把内容插入到文本框中内容的后面
#    #t.insert(1.1,var)  #把内容插入到第1行中的第1列
#    
#
#b1 = tk.Button(window,text = 'insert point',width = 15,height = 2,command = insert_point) #把内容插入到文本框中指针指定的位置
#b1.pack()
#b2 = tk.Button(window,text = 'insert end',command = insert_end)  #把内容插入到文本框的最后
#b2.pack()
#
#t = tk.Text(window,height = 2)   #声明一个文本框
#t.pack()   
#window.mainloop() #若缺少此句，则会造成窗口不会显示
    
    
######listbox控件的使用
#import tkinter as tk
#
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#
#var1 = tk.StringVar() #定义一个变量
#l = tk.Label(window,bg='yellow',width = 4,textvariable = var1)  #通过按钮触发为其设定值
#l.pack()
#
#def print_selection():
#    value = lb.get(lb.curselection())
#    var1.set(value)   #为第一个变量设值
#
#b1 = tk.Button(window,text='print selection',width = 15,height = 2,command = print_selection)
#b1.pack()
#    
#
#
#var2 = tk.StringVar()
##给变量var2设置一个初始值
#var2.set((11,22,33,44))
#lb = tk.Listbox(window,listvariable = var2)  #申明一个listbox，并把参数var2放入到里面:即给它赋初始值
#list_items = [1,2,3,4]
#for item in list_items:
#    lb.insert('end',item)   #往list中插入内容
#    
#lb.insert(1,'first')  #在索引为1处插入'first'
#lb.insert(2,'second')
#lb.delete(2)  #删除索引为2处的值
#
#lb.pack()
#window.mainloop()   


########radio按钮（单选按钮）
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#var = tk.StringVar()
#l = tk.Label(window,bg='yellow',width = 10,text = 'empty')   #显示所选的内容
#l.pack()
#
#def print_selection():
#    l.config(text = 'you have selected' + var.get())   #更改l标签中显示的内容
#
#
##这是一个radio按钮，上面显示的文字是'Option A',当选中时，将会给var这个参数赋值为‘A’
#r1 = tk.Radiobutton(window,text = 'Option A',variable = var,value = 'A',command = print_selection)
#r1.pack()
#
#r2 = tk.Radiobutton(window,text = 'Option B',variable = var,value = 'B',command = print_selection)
#r2.pack()
#
#r3 = tk.Radiobutton(window,text = 'Option C',variable = var,value = 'C',command = print_selection)
#r3.pack()
#
#window.mainloop()


######复选按钮checkButton
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#
#l = tk.Label(window,bg='yellow',width = 20,text ='empty')
#l.pack()
#
#def print_selection():
#    if (var1.get()==1)&(var2.get()==0):
#        l.config(text='I love only Python')
#    elif (var1.get()==0)&(var2.get()==1):
#        l.config(text = 'I love only C++')
#    elif (var1.get()==0)&(var2.get()==0):
#        l.config(text = 'I do not love either')
#    else:
#        l.config(text = 'I love both')
#        
#var1 = tk.IntVar()   #整数的变量
#var2 = tk.IntVar()
##var1 = tk.StringVar()  #字符串变量
#c1 = tk.Checkbutton(window,text ='Python',variable=var1,onvalue=1,offvalue=0,
#                        command = print_selection)   #variable表示哪个参数为其赋值，onvalue表示选中该按钮选中赋值为1（整数），不选中赋值为0
#c1.pack()
#c2 = tk.Checkbutton(window,text ='C++',variable=var2,onvalue=1,offvalue=0,
#                        command = print_selection)
#c2.pack()
#
#window.mainloop()



#######scale（尺寸）的用法
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#var = tk.StringVar()
#l = tk.Label(window,bg= 'yellow',width = 20,text = 'empty')
#l.pack()
#
#def print_selection(v):
#    l.config(text = 'you have selected'+v)   #v就是我们在滑动条中选中的值
#    
##showvalue=0表示在旁边不显示选中的值，若为1表示显示选中的值,resolution=0.01，表示保留以0.01为单位
#s = tk.Scale(window,label = "try me",from_ = 5,to_ = 11,orient = tk.HORIZONTAL,length = 200,showvalue = 0,
#             tickinterval = 3,resolution = 0.01,command = print_selection)  #定义一个横向的滑动条
#s.pack()
#window.mainloop()




######画布（canvas）
#import tkinter as tk 
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#canvas = tk.Canvas(window,bg = 'blue',height = 100,width = 200)   #创建一个画布
#image_file = tk.PhotoImage(file='ins.gif')       #往画布中加载一个图片的路径
##将图片加载到画布中
#image = canvas.create_image(10,10,anchor='nw',image = image_file)   #在西北位置处，顶点为（0，0），然后找（10，10）作为图片的起点
#
##自己创建不同形状的物体
#x0,y0,x1,y1 = 50,50,80,80  #定义两个点的坐标
#line = canvas.create_line(x0,y0,x1,y1)    #创建一条直线
#oval = canvas.create_oval(x0,y0,x1,y1,fill = 'red')  #创建一个圆
#arc = canvas.create_arc(x0+30,y0+30,x1+30,y1+30,start = 0,extent = 180) #创建一个扇形，有角度的起点，还有角度的终点
#rect = canvas.create_rectangle(100,30,100+20,30+20)  #创建一个矩形
#canvas.pack()
#
#def moveit():
#    canvas.move(rect,0,2)  #正方形每一次向下移动单位2（0表示x轴移动为0，y轴表示移动单位为2）
#    
#b = tk.Button(window,text ='move',command = moveit)  #点击这个按钮时，矩形会发生移动
#b.pack()
#
#window.mainloop()



#######Menubar(菜单)
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#l = tk.Label(window,text ='',bg = 'yellow')
#l.pack()
#
#counter = 0
#def do_job():
#    global counter
#    l.config(text = 'do'+str(counter))
#    counter +=1
#
#menubar = tk.Menu(window)  #创建一个menubar
#
#
#filemenu = tk.Menu(menubar,tearoff = 0)  #定义一个文件菜单（filemenu）
#menubar.add_cascade(label = 'File',menu = filemenu)    #将文件菜单添加到总菜单中，并给这个菜单命名为‘File’
##给文件菜单定义子菜单
#filemenu.add_command(label = 'New',command = do_job)  #在文件菜单下定义一个子菜单名称为‘New’
#filemenu.add_command(label = 'Open',command = do_job)
#filemenu.add_command(label = 'Save',command = do_job)
#filemenu.add_separator()    #添加一个分割符,即一条直线
#filemenu.add_command(label = 'Exit',command = window.quit)
#
#
#editmenu = tk.Menu(menubar,tearoff = 0)  
#menubar.add_cascade(label = 'Edit',menu = editmenu)    #将文件菜单添加到总菜单中，并给这个菜单命名为‘File’
##给文件菜单定义子菜单
#editmenu.add_command(label = 'Cut',command = do_job)  #在文件菜单下定义一个子菜单名称为‘New’
#editmenu.add_command(label = 'Copy',command = do_job)
#editmenu.add_command(label = 'Paste',command = do_job)
#
#submenu = tk.Menu(filemenu)  #在文件菜单中建立一个子菜单
#filemenu.add_cascade(label = 'Import',menu=submenu,underline = 0)
#submenu.add_command(label='submenu1',command = do_job)  #在子菜单中添加一个子菜单
#window.config(menu=menubar)
#window.mainloop()



#######Frame（框架）
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#l =tk.Label(window,text = 'on the window')   #创建一个标签,放在主frame
#l.pack()
#
#
#frm = tk.Frame(window)  #往窗口中添加一个主框架（frame）
#frm.pack()
##在框架的左边添加一个子框架
#frm_l = tk.Frame(frm)
#frm_l.pack(side = 'left')
##在框架的右边添加一个子框架
#frm_r = tk.Frame(frm)
#frm_r.pack(side = 'right')
#
##以下两个标签放在左子frame中
#tk.Label(frm_l,text ='on the frm_l1').pack()  
#tk.Label(frm_l,text = 'on the frm_l2').pack()
#
##放到右子frame中
#tk.Label(frm_r,text = 'on the frm_r1').pack()
#
#window.mainloop()



#######messagebox(弹窗)
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#def hit_me():
#    
#    #tk.messagebox.showinfo(title = 'Hi',message = 'hahahaha')   #提示信息
#    #tk.messagebox.showwarning(title ='Hi',message = 'nononono')    #警告信息
#    #tk.messagebox.showerror(title = 'Hi',message = 'no!never')
#    #print(tk.messagebox.askquestion(title = 'Hi',message ='hahahaha'))   #return 'yes'  or 'no'
#    #print(tk.messagebox.askyesno(title = 'Hi',message = 'hahahah'))      #return True,False
#    print(tk.messagebox.askokcancel(title='hi',message = 'hahaha'))      #return True,False
#button = tk.Button(window,text = 'hit me',command = hit_me)
#button.pack()
#window.mainloop()




########放置控件的几种方法：place()  grid()（网格形式）   place()的对比
#import tkinter as tk
#window = tk.Tk()
#window.title('my window')
#window.geometry('200x200')
#
#####pack()的使用
###生成4个标签，text是这个label上显示的内容
##tk.Label(window,text = 'top').pack(side ='top')   
##tk.Label(window,text ='bottom').pack(side='bottom')
##tk.Label(window,text = 'left').pack(side = 'left')
##tk.Label(window,text = 'right').pack(side = 'right')
#
#
#####grid():网格布局的使用
##for i in range(4):
##    for j in range(3):
##        #row和column是控制当前标签放置的位置，，，，padx和pady是控制当前控制放置位置的大小
##        tk.Label(window,text = 1).grid(row = i,column = j,padx = 5,pady = 1)  
#
#
######place()的使用
#tk.Label(window,text = 1).place(x=10,y=100,anchor = 'nw')  #nw表示以西北角为（0，0）起点，(x,y)表示该标签所放置的位置
#
#
#window.mainloop()

#
######小案例：登录窗口的制作
#import tkinter as tk
#import pickle  #需要先打开文件，再进行读/写操作
#window = tk.Tk()
#window.title('my window')
#window.geometry('450x300')   #window窗口的大小
#
##加载图片
#canvas = tk.Canvas(window,height = 200,width = 500)    #创建一个画布
#image_file = tk.PhotoImage(file = 'welcome.gif')    #要放置到画布上的图片
#image = canvas.create_image(0,0,anchor ='nw',image = image_file)   #把图片放置到我们设定的位置上
#canvas.pack(side='top')   #规定画布放置到窗口的位置上
#
##设定输入信息的布局，以及对输入信息进行控制
#tk.Label(window,text = 'User name:').place(x=50,y = 150)
#tk.Label(window,text = 'Password').place(x=50,y= 190)
#
#var_usr_name = tk.StringVar()
#var_usr_name.set('example@python.com')  #给这个变量一个默认值，即是提示信息
#var_usr_pwd = tk.StringVar()
#
#entry_usr_name = tk.Entry(window,textvariable = var_usr_name)   #设定输入用户名信息框
#entry_usr_name.place(x=160,y=150)
#entry_usr_password = tk.Entry(window,textvariable = var_usr_pwd,show = '*')   #输入的信息以‘*‘显示
#entry_usr_password.place(x= 160,y=190)
#
#
##定义登录的函数
#def usr_login():
#    usr_name = var_usr_name.get()
#    usr_pwd = var_usr_pwd.get()
#    try:
#        with open('usrs_info.pickle','rb') as usr_file:   #打开usrs_info.pickle文件的信息
#            usrs_info = pickle.load(usr_file)  #读取里面的内容
#        except FileNotFoundError:
#            with open('user_info.pickle','wb') as usr_file:
#                usrs_info = {'admin':'admin'}  #获得管理员的信息，把其信息写入到文件中共
#                pickle.dump(usrs_info,usr_file)   #若文件内容不存在，则把管理员信息写进去，之后以管理员的身份进行登录
#        if usr_name in usrs_info:
#            if usr_pwd == usrs_info[usr_name]:
#                tk.messagebox.showinfo(title='welcome',message = 'How are you?' + usr_name)   #提示信息：登录正确
#            else:
#                tk.messagebox.showerror(message = 'Error,your password is wrong,try again')  #密码错误
#        else:  #当前用户名不存在，提示去注册
#            is_sign_up = tk.messagebox.askyesno(title = 'welcome',message = 'you have not sign uo yet, Sign up')
#       
#        
#        
#def usr_sign_up():
#    
#    #定义一个函数
#    def sign_to_python():
#        
#        ##得到3个输入框的内容
#        np = new_pwd.get()
#        npf = new_pwd_confirm.get()
#        nn = new_name.get()
#        with open('usr_info.pickle','rb') as usr_file:
#            exist_usr_info = pickle.load(usr_file)    #加载保护用户的信息
#        
#        #判断密码和用户的确认密码是否一致
#        if np! =npf:
#            tk.messagebox.showerror('Error','password and confirm password must be the same')
#        elif nn in exist_usr_info:  #注册的用户信息已经存在
#            tk.messagebox.showerror('Error','the user has exist')
#        else:  #注册合格
#            exist_usr_info[nn] =np  #把信息保存到文件中
#            with open('usrs_info.pickle','wb') as usr_file:
#                pickle.dump(exist_usr_info,usr_file)
#            tk.messagebox.showinfo('Welcome','you have successfully signed up')
#            window_sign_up.destroy()  #关闭这个窗口
#        
#
#    ####注册窗口
#    window_sign_up = tk.Toplevel(window)   #建立一个toplevel形式的window
#    window_sign_up.geometry('350x200')
#    window_sign_up.title('Sign up window')
#    
#    new_name = tk.StringVar()      #申明一个参数
#    new_name.set('example@python.com')  #给这个变量设置初始值
#    tk.Label(window_sign_up,text = 'User name:').place(x=10,y = 10)   #在这个window中放一个标签，text是标签上显示的内容
#    entry_new_name = tk.Entry(window_sign_up,textvariable = new_name)   #声明一个输入框，并给它一个默认值
#    entry_new_name.place(x= 150,y = 10)   #设定这个输入框放置的位置
#    
#    #注册的密码
#    new_pws = tk.StringVar()
#    tk.Label(window_sign_up,text = 'Password:').place(x= 10,y=50)
#    entry_usr_pwd = tk.Entry(window_sign_up,textvariable)
#    entry_usr_pwd.place(x=150,y = 50)
#    
#    #确认注册的密码
#    new_pwd_confirm = tk.StringVar()
#    tk.Label(window_sign_up,text= 'confirm password:').place(x=10,y = 90)
#    entry_usr_pwd_confirm = tk.Entry(window_sign_up,textvariable = new_pwd_confirm,show = '*')
#    entry_usr_pwd_confirm.place(x=150,y=90)
#    
#    
#    mfirm_sign_up = tk.Button(window_sign_up,text = 'sign up',command = sign_to_python)   #申明一个按钮
#    mfirm_sign_up.place(x= 150,y = 130)
#    
#    
#    
#    
#
##登录和注册按钮
#btn_login = tk.Button(window,text = 'Login',command = usr_login)
#btn_login.place(x=170,y=230)
#btn_sign_up = tk.Button(window,text = 'sign up',command = usr_sign_up)
#btn_sign_up.place(x=270,y= 230)
#
#
#window.mainloop()












    








