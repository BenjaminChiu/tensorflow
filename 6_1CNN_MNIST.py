import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
# one_hot 指将图片转换成一维向量，如标签0 表示 ([1,0,0,0,0,0,0])，1与其他8个0，共9位
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次，整除 下取整
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    # reduce他妈的什么意思？
    mean = tf.reduce_mean(var)








# 定义两个placeholder，接受传入的样本集
# x 代表 这个批次中某个字符，x 介于0到99，共一百个数
x = tf.placeholder(tf.float32, [None, 784])
# y 代表 传入的数据集的 正确标签
y = tf.placeholder(tf.float32, [None, 10])
# 改进2_1：多于这个用来接受drop_out，避免过拟合；整个改进2都是以dropout为核心的改动
keep_prob = tf.placeholder(tf.float32)
# 改进4： 学习率变量
lr = tf.Variable(0.001, dtype=tf.float32)


# 创建一个简单的神经网络
# 变量初始化为0；784 来自x；10来自y
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))




# 改进2_2：使用截断的正态分布来初始化，标准差0.1
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# 偏置值也做相应修改
b1 = tf.Variable(tf.zeros([500])+0.1)
# 每一个神经元层的输出 L1
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
L1_drop = tf.nn.dropout(L1, keep_prob)

# 第二个神经元层
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
# 偏置值也做相应修改
b2 = tf.Variable(tf.zeros([300])+0.1)
# 每一个神经元层的输出 L1
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
L2_drop = tf.nn.dropout(L2, keep_prob)

#
# # 第3个神经元 层
# W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# # 偏置值也做相应修改
# b3 = tf.Variable(tf.zeros([10])+0.1)
# # 每一个神经元层的输出 L1
# L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# L3_drop = tf.nn.dropout(L3, keep_prob)

# 第4个神经元层 输出层
W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# 偏置值也做相应修改
b4 = tf.Variable(tf.zeros([10])+0.1)
# 每一个神经元层的输出 L1
# L4 = tf.nn.tanh(tf.matmul(L2_drop, W4) + b4)
# # 将输出结果 交给dropout，keep prob是神经元的百分比，取值范围0~1
# L4_drop = tf.nn.dropout(L2, keep_prob)
# 改进2_2===结束=====


prediction = tf.nn.softmax(tf.matmul(L2_drop, W4) + b4)



# 二次 代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))
# 改进1_1：启用交叉熵 代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))


# 优化器：使用梯度下降法，学习率0.2；神TM0.2 怎么来的？
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 改进3：换装优化器；1e-2 表示10的-2次方
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


# 使用了变量，初始化变量
init = tf.global_variables_initializer()


# 结果存放在一个布尔型列表中
# argmax返回一维张量中最大的值所在的位置？？
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 训练21个周期
    for epoch in range(50):
        # 学习率learning read，让学习率缓慢下降，前期快速靠近局部最小值，后期缓慢靠近
        # 类似过山车下坡时，慢慢踩刹车，才能准确停靠在最低点，不至于冲上另一边山头
        sess.run(tf.assign(lr, 0.001*(0.95**epoch)))

        # 一个批次的训练
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        learning_rate = sess.run(lr)
        # 一个批次的测试
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})



        # print("当前训练周期 " + str(epoch) + "，测试准确率 " + str(test_acc)+"，训练集准确率：" + str(train_acc))
        print("当前训练周期 " + str(epoch) + "，测试准确率 " + str(test_acc)+"，学习率：" + str(learning_rate))
