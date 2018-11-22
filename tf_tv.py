# '''使用tf实现tv最小化
# 适用于处理一维序列数据
# tf、np都是约定HW，通常说的800X600都是高度600、宽度800
# 111'''
# 
# a = np.array(np.arange(1, 1 + 20).reshape([1, 10, 2]), dtype=np.float32)  # 1个样本，每个样本10x2
# # 卷积核，此处卷积核的数目为1
# 有batch。有通道！！！！！
# kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1]) 
# 单一卷积核为2大小，处理输入“通道2”，1个卷积核
# # 进行conv1d卷积
# conv1d = tf.nn.conv1d(a, kernel, 1, 'VALID')
# with tf.Session() as sess:
#    # 初始化
#    tf.global_variables_initializer().run()
#    # 输出卷积值
#    print(sess.run(conv1d))
#
# 若是差分向量求norm，则是L2
# 若是查分向量分量p次再求和、求p次方根，则是Lp
# 1次就是分量的原始数字，所以直接相加就是l1，一次根也是原始数
# 显然不会有L0，因为总会有不同，而不是这个地方不变，缺乏表征性

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def TV_LOSS_1d_L1_NP(x, y, v_lambda):
  '''np实现   数值计算OK
  成为单个函数，没有了内部函数的调用
  包括了两个代价成分的最小化目标
  向量的MSE以及lambda加权的TV值
  x   原始序列
  y   重建序列
  v_lambda  控制参数，0没有过滤，无穷是极度过滤
  两个向量相同长度'''
  return np.linalg.norm(x-y) + v_lambda*np.sum( np.absolute( np.ediff1d( y,to_begin=0 ) ) ) 


def TV_LOSS_1d_L1(x,yy, v_lambda):
  '''使用tf实现
  由于一维卷积实现细节的约束，一维向量在送入模型的时候需要变成3d张量【B，N，C】
  网络输出yy是形状为【b n c】的张量，而不是一维
  但是输入矢量x却是1维，所以需要reshape   不能直接相减，否则发生广播
  约定batch=1、chennal = 1，否则出错
  或者约定，输入x是网络的输入，或是具有对应形状的张量
  '''

  # 方案，使用常量卷积实现差分，也就是相减
  diff = tf.constant([1.,-1.], tf.float32, shape=[2,1,1])
  # 卷积核尺寸和表征张量顺序不同，这是【h,w,inC,outC】 这里将会自动变为【1，2，1，1】                      
  diff_v = tf.nn.conv1d(yy, diff, 1,'SAME')
  # 数据为什么是【1，1，1000，1】形状？batch必须有  自身将一维转换成为2维、含有深度？？
  # 这一行出错 got shape [7], but wanted [] 这是因为传入的是含有所有层的“模型”而不是单一的层
  return tf.linalg.norm( x  - \
                         yy) \
         + \
         v_lambda*tf.reduce_sum( tf.abs( diff_v ) )


def TV_AE_C1d(s,v_lambda):
  '''
  使用3个卷积层，每个层使用两种不同尺度的卷积核
  只有1个样本的情况下，进行反复的训练，batch怎么处理?
  原始程序中，对于mnist数据作为numpy的array
  对于这个array自动进行（同步的）遍历
  输入序列虽然是一维的，但是似乎需要变成2维，n行1列的方式表示
  tf的一维卷积不是为了纯粹的信号处理设计的，是2d卷积的一种可替代形式，本质还是2d卷积

  因此处理数据不能是1d序列?????
  '''
  tf.get_default_graph()
  tf.reset_default_graph()
  
  sess = tf.InteractiveSession()

  #s = s.reshape((1,-1))# 1d卷积的关系，需要变成2d描述
  #s = s.reshape((-1,1))# 数据是行还是列，对于后续
  data_shape = tf.shape(s)
  ndims = len(np.shape(s))
  if ndims != 1:
    raise ValueError(r"输入数据需要是一维向量")
  # 序列数据，形状【L】
  # 图像，形状【h,w,c】或是【h,w】
  final_filters =  1# 未来兼容多通道图像的目的
  feed_data = np.reshape(s, (1,s.size,1)) #np.tile(s, (1,1))# 增加一个维度表示batch,一个维度表示“通道”
  feed_data = feed_data.astype(np.float32)
  # placeholder
  x = tf.placeholder(  tf.float32, 
                       shape=[*tf.shape(feed_data).eval()] # 这里怎么突然出错
                       )# batch==1
  # shape=[]或是shape=[None,None]也可以
  print(x)
  # tl.layers.set_name_reuse(True)
  print("\n Build net--------------")# -----------------------------------------------
  input_layer = tl.layers.InputLayer(x, name='input')
  cnn1_1 = tl.layers.Conv1d(input_layer, 16, 3, 1, 
                              act=tf.nn.relu, 
                           padding='SAME', name='cnn1_1')# 
  cnn1_2 = tl.layers.Conv1d(input_layer, 16, 5, 1, 
                              act=tf.nn.relu, 
                           padding='SAME', name='cnn1_2')#   # 第二个相同的输入，似乎就冲突了？！
  cnn_1 = tl.layers.ConcatLayer([cnn1_1,cnn1_2])#,name='cnn_1'

  cnn2_1 = tl.layers.Conv1d(cnn_1, 16, 3, 1, 
                              act=tf.nn.relu, 
                           padding='SAME', name='cnn2_1')
  cnn2_2 = tl.layers.Conv1d(cnn_1, 16, 5, 1, 
                              act=tf.nn.relu, 
                           padding='SAME', name='cnn2_2')
  cnn_2 = tl.layers.ConcatLayer([cnn2_1,cnn2_2],
                                  name='cnn_2')

  y = tl.layers.Conv1d(cnn_2, final_filters, 3, 1, 
                         act=tf.nn.relu, 
                           padding='SAME', name='out')

  # ready to train
  tl.layers.initialize_global_variables(sess)

  # print all params
  print("网络参数");  y.print_params()
  print("网络层");  y.print_layers()

  print("进行拟合")
  cost= TV_LOSS_1d_L1(x, # 这里输入同型的张量  不能使用inputlayer，类型问题
                      y.all_layers[-1], 
                      v_lambda)# 
  #为什么得到shape7，而不是期望的长度？似乎因为y不是这个张量，而是这个网络？！？！7是各个层
  # 所以使用all_layers[-1]得到最后的激活层，

  # optimizer = tl.optimizers.AMSGrad()
  optimizer = tf.train.GradientDescentOptimizer(0.3)
  optimizer = optimizer.minimize(cost)
  # 需要加入学习率的控制
  
    
  for _ in range(10):# 这么多次拟合、游走。当心步长的不合适
    sess.run(optimizer,
             feed_dict={x:feed_data}# builtins.TypeError: input must be a dictionary
            # got multiple values for argument 'feed_dict'cost,
                 )
    print(cost.eval())

  #out_put = y.all_layers[-1].eval()#将张量转换成为numpy数据  对应还有 tf.convert_to_tensor()
  out_put = tf.reshape(y.all_layers[-1], data_shape, name=None)
  #out_put = tf.convert_to_tensor(out_put)
  out_put = out_put.eval(feed_dict={x:feed_data})
  sess.close()

  return out_put


def _blockysignal_1d():
  """
  前1/4    0
  又1/4    1
  又1/4    -1
  又1/8    2
  末1/8    0

  |            2
  |    1 1
  |0 0            
  |       -1 -1   0 
  1 2 3 4  5  6 7 8
  """
  N = 1000
  s = np.zeros(N)# 一维向量序列？
  #s = np.zeros((N,1))# 2维单列序列？ 看做N个采样，每个采样1特征
  #s = np.zeros((1,N))# 2维单行序列？ 看做一个样本，每个特征N个  不显示
  s[int(N/4):int(N/2)] = 1
  s[int(N/2):int(3*N/4)] = -1
  s[int(3*N/4):int(-N/8)] = 2
  return s

def _do_test():
  # 会经常用到
  # import importlib
  # from tf_tv import *;
  # import tf_tv
  # tf_tv._do_test()
  # importlib.reload(tf_tv)
  s = _blockysignal_1d()# 方波信号
  sn = s + np.random.random_sample(s.shape)-.5# 均值0、取值+——1的噪声
  sr = TV_AE_C1d(s, 0)# 重建信号，lambda=5
  # 结果的绘制
  import matplotlib.pyplot as mpl
  # 绘制数据不能是行。
  # 一行数据却作为一个N维样本，不显示
  # 一个单列数据看做N个采样
  # 或者转换成为一维数据，没问题
  mpl.plot(sn)
  mpl.plot(s)
  mpl.plot(sr)
  mpl.show()
