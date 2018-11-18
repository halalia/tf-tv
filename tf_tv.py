'''
使用tf实现tv最小化
适用于处理一维序列数据
tf、np都是约定HW，通常说的800X600都是高度600、宽度800
'''
import tensorflow as tf
import tensorlayer as tl
import numpy as np

def TV_LOSS_1d_L1_NP(x, y, v_lambda):
    '''
    np实现   数值计算OK
    成为单个函数，没有了内部函数的调用
    包括了两个代价成分的最小化目标
    向量的MSE以及lambda加权的TV值
    x   原始序列
    y   重建序列
    v_lambda  控制参数，0没有过滤，无穷是极度过滤
    两个向量相同长度
    '''
    return np.linalg.norm(x-y) + v_lambda*np.sum( np.absolute( np.ediff1d( y,to_begin=0 ) ) ) 


def TV_LOSS_1d_L1(x,y, v_lambda):
    '''使用tf实现
    np.linalg.norm    tf.linalg.norm
    np.sum            tf.reduce_sum
    np.absolute       tf.abs
    np.ediff1d        没有对应  使用两个张量（矢量）相减
    
    '''
    #v_size = tf.shape(y)# 传播之后得到shape为7？？？？？got shape [7], but wanted [].
    #print(v_size,   tf.size(x))
    #diff_v = tf.slice(y,0,v_size-1)  - tf.slice(y,1,v_size-1)
    # 即使尺寸相差1，但是对于求值的影响忽略
    
    # 方案2，使用常量卷积实现差分，也就是相减
    #diff = tf.reshape(tf.constant([1.,-1.],
                                  #tf.float32,
                                  #shape=[1,2,1]),
                      #)
    diff = tf.constant([1.,-1.],
                       tf.float32,
                       shape=[2,1,1])# 卷积核尺寸和表征张量不同，这是【h,w,inC,outC】                      
    diff_v = tf.nn.conv1d(y.all_layers[-1],# 数据为什么是【1，1，1000，1】形状？batch必须有  自身将一维转换成为2维、含有深度？？
                          diff,1,'SAME')
    # 这一行出错 got shape [7], but wanted []
    return tf.linalg.norm(x-y.all_layers[-1]) + v_lambda*tf.reduce_sum( tf.abs( diff_v ) )


def TV_AE_C(din,v_lambda):
    '''
    使用3个卷积层，每个层使用两种不同尺度的卷积核
    只有1个样本的情况下，进行反复的训练，batch怎么处理?
    原始程序中，对于mnist数据作为numpy的array
    对于这个array自动进行（同步的）遍历
    '''
    sess = tf.InteractiveSession()
    data_shape = tf.shape(din)
    dims = len(np.shape(din))
    # 序列数据，形状【L】
    # 图像，形状【h,w,c】或是【h,w】
    final_filters = data_shape[-1] if dims==3 else 1
    # placeholder
    x = tf.placeholder(tf.float32, 
                       #shape=tf.shape(tf.expand_dims(din, 0)).eval(session=sess), 
                       shape = [],
                       name='input')# batch==1
    # shape=[]或是shape=[None,None]也可以
    print("Build net")

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
    print("网络参数")
    y.print_params()
    print("网络层")
    y.print_layers()

    # pretrain
    print("进行拟合")
    cost= TV_LOSS_1d_L1(x, y, v_lambda)# 
    #为什么得到shape7，而不是期望的长度？似乎因为y不是这个张量，而是这个网络？！？！7是各个层

    # optimizer = tl.optimizers.AMSGrad()
    optimizer = tf.train.GradientDescentOptimizer(0.3)
    optimizer = optimizer.minimize(cost)
    for _ in range(10):# 这么多次拟合、游走。当心步长的不合适
        sess.run(cost,
                 optimizer,
                 feed_dict={x:din})
        
    out_put = out.eval(session=sess)#将张量转换成为numpy数据  对应还有 tf.convert_to_tensor()
    sess.close()
    
    return out_put



def _blockysignal():
    """
    前1/4    0
    又1/4    1
    又1/4    -1
    又1/8    2
    末1/8    0
    
                2
        1 1
    0 0            
           -1 -1   0 
    1 2 3 4  5  6 7 8
    """
    N = 1000
    s = np.zeros((N,1))
    s[int(N/4):int(N/2)] = 1
    s[int(N/2):int(3*N/4)] = -1
    s[int(3*N/4):int(-N/8)] = 2
    return s

def _do_test():
    # 会经常用到
    # import importlib
    # import tf_tv
    # tf_tv._do_test()
    # importlib.reload(tf_tv)
    s = _blockysignal()# 方波信号
    sn = s+np.random.random((np.size(s),1))-.5# 均值0、取值+——1的噪声
    sr = TV_AE_C(s, 5)# 重建信号，lambda=5
    # 结果的绘制
    import matplotlib.pyplot as mpl
    mpl.plot(s)
    mpl.plot(sn)
    mpl.plot(sr)
    mpl.show()
