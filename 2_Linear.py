#简单的线性模型

import tensorflow as tf
import numpy as np

# 样本：使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*5 + 8

# 自己构造一个线性模型 来 模拟上面现有的样本
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 差距：二次代价函数，即获取模型和样本的 差的平方
loss = tf.reduce_mean(tf.square(y_data-y))

# 优化器：定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 用优化器来不断减小差距，最小化代价函数
train = optimizer.minimize(loss)


#使用了变量，初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  #初始化 op
    for step in range(201):
        sess.run(train)     #训练 OP
        if step % 20 == 0:
            print(step, sess.run([k, b]))