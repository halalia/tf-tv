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

def TV_AE_SFC(model='relu'):
    # 堆叠的全连接
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    if model == 'relu':
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif model == 'sigmoid':
        act = tf.nn.sigmoid
        act_recon = act

    # Define net
    print("\nBuild net")
    net = tl.layers.InputLayer(x, name='input')
    # denoise layer for AE
    net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')
    # 1st layer
    net = tl.layers.DropoutLayer(net, keep=0.8, name='drop1')
    net = tl.layers.DenseLayer(net, n_units=800, act=act, name=model + '1')
    x_recon1 = net.outputs
    recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=act_recon, name='recon_layer1')
    # 2nd layer
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, n_units=800, act=act, name=model + '2')
    recon_layer2 = tl.layers.ReconLayer(net, x_recon=x_recon1, n_units=800, act=act_recon, name='recon_layer2')
    # 3rd layer
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
    net = tl.layers.DenseLayer(net, 10, act=None, name='output')

    # Define fine-tune process
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = net.all_params

    # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

    # Initialize all variables including weights, biases and the variables in train_op
    tl.layers.initialize_global_variables(sess)

    # Pre-train
    print("\nAll net Params before pre-train")
    net.print_params()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(
        sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10,
        save=True, save_name='w1pre_'
    )
    print("\nPre-train Layer 2")
    recon_layer2.pretrain(
        sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10,
        save=False
    )
    print("\nAll net Params after pre-train")
    net.print_params()

    # Fine-tune
    print("\nFine-tune net")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1  # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     # visualize the 1st hidden layer during fine-tune
            #     tl.vis.draw_weights(net.all_params[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))
    # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def TV_AE_FC(model='relu'):

    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

    print("Build net")
    if model == 'relu':
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')  
        # if drop some inputs, it is denoise AE
        net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.relu, name='relu1')
        recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.softplus, name='recon_layer1')
    elif model == 'sigmoid':
        # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')
        net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.sigmoid, name='sigmoid1')
        recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')

    # ready to train
    tl.layers.initialize_global_variables(sess)

    # print all params
    print("All net Params")
    net.print_params()

    # pretrain
    print("Pre-train Layer 1")
    recon_layer1.pretrain(
        sess, 
        x=x, 
        X_train=X_train, 
        X_val=X_val, 
        denoise_name='denoising1', 
        n_epoch=200, batch_size=128, print_freq=10,
        save=True, save_name='w1pre_'
        )
    # You can also disable denoisong by setting denoise_name=None.
    # recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    #                           denoise_name=None, n_epoch=500, batch_size=128,
    #                           print_freq=10, save=True, save_name='w1pre_')

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

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