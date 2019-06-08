import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
# one_hot 指将图片转换成一维向量，如标签0 表示 ([1,0,0,0,0,0,0])，1与其他8个0，共9位
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次，整除 下取整
n_batch = mnist.train.num_examples // batch_size



# 初始化权值
# shape为4维tensor，各参数含义：[width, height, channels, kernel_nums]
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    #生成一个截断的正态分布；stddev是标准差(Standard Deviation)
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积操作，卷积层
def conv2d(x, W):

    # 参数x 表示：an input tensor of shape `[batch, in_height, in_width, in_channels]`
    #                     tensor变量         批次     长           宽      通道数/频道（黑白通道为1，彩色通道为3 rgb）

    # 参数W 表示：a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
    #             卷积核/滤波器 功能的 tensor变量        卷积核的长         宽      卷积核输入通道数  输出通道数

    # strides步长： Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    #   horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    # strides[1]表示X方向的步长（横向X坐标轴），strides[2] 表示Y方向的步长

    # padding: A `string` from: `"SAME", "VALID"`.
    # SAME 即得到的结果矩阵 与 卷积矩阵 大小相同（长宽相同）
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    # ksize[1,x,y,1] 窗口大小 x表示x方向的大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 定义两个placeholder，接受传入的样本集
# x 代表 这个批次中某个字符，x 介于0到99，共一百个数
x = tf.placeholder(tf.float32, [None, 784])  # None表示行 图片当前批次总数 ,784 = 28*28转1维向量
# y 代表 传入的数据集的 正确标签
y = tf.placeholder(tf.float32, [None, 10])

# 复原x中 的图片
# 改变X的格式 转为4D的向量[batch， in_height,in_width, in_channels]
#                       批次大小    长           宽       通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 第一个卷积层的权值 和 偏置
# shape为4维tensor，各参数含义：[width, height, channels, kernel_nums]
W_conv1 = weight_variable([5, 5, 1, 32]) #5*5的卷积窗口，32个卷积核从1个通道抽取特征，一个卷积核 卷积操作后得到一个特征平面
b_conv1 = bias_variable([32])   #每一个卷积核 一个偏置值


# 把x_image和权值向量 进行卷积，再加上偏置值，应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)     #进行 maxpool池化


#=======================神经网络层==分割线===================

# 第二个卷积层的权值 和 偏置
W_conv2 = weight_variable([5, 5, 32, 64]) #5*5的卷积窗口，64个卷积核从32个平面抽取特征
b_conv2 = bias_variable([64])   #每一个卷积核 一个偏置值
# 为什么一个卷积核配一个偏置值？ 个人理解：在以前全连接网络中，一个像素点对应一个神经元
# 那在卷积层中 就是 一个大的神经元（卷积核） 对应 一大片输入像素，这个大的神经元 配一个偏置值是合理的


# 把x_image和权值向量 进行卷积，再加上偏置值，应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)     #进行 maxpool池化



# 28*28的图片 第一次卷积后还是 28*28，第一次池化后变为14*14
# 第二次卷积后是14*14，第二次池化后变为7*7
# 进行过上面操作后，得到64张7*7的平面


#初始化 第一个全连接层的权值，全连接层为什么也要使用这个自定义的初始化方案
W_fcl = weight_variable([7*7*64, 1024])         #上一层有7*7*64个神经元，全连接层有1024个神经元
b_fcl = bias_variable([1024])       #1024个节点

#把池化层2的输出 扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 求第一个全连接的输出
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)

# keep_prob 用来表示神经元输出的概率
keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)



# ===== 第二个全连接层 用于分类===========================
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])



# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)


# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 将结果存放在一个布尔列表中 argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#求准确率
accuracy = accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))














# # 改进2_2：使用截断的正态分布来初始化，标准差0.1
# W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# # 偏置值也做相应修改
# b1 = tf.Variable(tf.zeros([500])+0.1)
# # 每一个神经元层的输出 L1
# L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# # 第二个神经元层
# W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
# # 偏置值也做相应修改
# b2 = tf.Variable(tf.zeros([300])+0.1)
# # 每一个神经元层的输出 L1
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# #
# # # 第3个神经元 层
# # W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# # # 偏置值也做相应修改
# # b3 = tf.Variable(tf.zeros([10])+0.1)
# # # 每一个神经元层的输出 L1
# # L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# # # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# # L3_drop = tf.nn.dropout(L3, keep_prob)
#
# # 第4个神经元层 输出层
# W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# # 偏置值也做相应修改
# b4 = tf.Variable(tf.zeros([10])+0.1)
# # 每一个神经元层的输出 L1
# # L4 = tf.nn.tanh(tf.matmul(L2_drop, W4) + b4)
# # # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# # L4_drop = tf.nn.dropout(L2, keep_prob)
# # 改进2_2===结束=====
#
#
# prediction = tf.nn.softmax(tf.matmul(L2_drop, W4) + b4)
#
#
#
# # 二次 代价函数
# # loss = tf.reduce_mean(tf.square(y - prediction))
# # 改进1_1：启用交叉熵 代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#
#
# # 优化器：使用梯度下降法，学习率0.2；神TM0.2 怎么来的？
# # train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# # 改进3：换装优化器；1e-2 表示10的-2次方
# train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#
#
# # 使用了变量，初始化变量
# init = tf.global_variables_initializer()
#
#
# # 结果存放在一个布尔型列表中
# # argmax返回一维张量中最大的值所在的位置？？
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#
# # 求准确率


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练21个周期
    for epoch in range(20):
        # 一个批次的训练
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        # 一个批次的测试
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})



        # print("当前训练周期 " + str(epoch) + "，测试准确率 " + str(test_acc)+"，训练集准确率：" + str(train_acc))
        print("当前训练周期 " + str(epoch) + "，测试准确率 " + str(test_acc))
