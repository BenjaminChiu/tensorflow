# 非线性回归的例子


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 样本：使用numpy生成200个随机点
# 如果把（-0.5,0.5,200）改成(-5,5,200)预测值就会全为1，因为超出了tanh值域
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)         #wtf x_data.shape???  为什么是0.2
y_data = np.square(x_data) + noise

# 输入层？
# 定义两个占位符
# 有的模型要数据要一块一块地传入，placeholder就很方便
# 感觉相当于占位占了无限个 由你自己定
# x第一维是batch的大小，第二维是特征数量，因为你每次batch大小可以是任意的，所以一般placeholder的第一维要写None

x = tf.placeholder(tf.float32, [None, 1])
# 训练阶段才会用到真实值y,预测阶段不能用真实值y，因为要用预测值与真实值进行比较
# 训练好了，实际输出值Ｙ就没用了，y属于测试集 不参与训练
# 做梯度下降的训练过程才需要用到真实的y把参数训练出来,这个过程已经结束了。现在把x带进去就可以得到prediction了
y = tf.placeholder(tf.float32, [None, 1])

# 神经网络就是矩阵的计算，把矩阵的shape与一些定义如神经元，特征数什么结合起来理解

#定义神经网络中间层

Weights_L1 = tf.Variable(tf.random_normal([1, 10]))     #先随机赋值；权值 连接 输入层与中间层（输入层1个神经元，中间层10个神经元）
biases_L1 = tf.Variable(tf.zeros([1, 10]))              #中间层10个神经元，第二个参数10，第一个参数是什么意思？？？一个输入层神经元？
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1     #只是信号的相加，变量的相加，不是矩阵的相加
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


# 差距：二次代价函数，即获取模型和样本的 差的平方
loss = tf.reduce_mean(tf.square(y-prediction))

# 优化器：定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)    # 0.1 ????

# 用优化器来不断减小差距，最小化代价函数
train = optimizer.minimize(loss)


# ========================
# Wx_plus_b_L1 信号的总和 =
#
# ==样本/输入 (矩阵)  *  权值（矩阵）+偏置值
#
# sess x:x_data  将x_data穿给x 当做传入的数据
#
# 梯度下降法中的 参数意义
# ========================



#使用了变量，初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  #初始化变量 op

    i=0
    #训练得到模型
    for _ in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})     # 训练 OP

    #用模型 获得预测值
    prediction_value = sess.run(prediction,feed_dict={x: x_data})

    #画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=3)      #红色实线，线宽5
    plt.show()