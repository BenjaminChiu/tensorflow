import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

c = tf.constant('Hello, World!')
with tf.Session() as sess:
    print(sess.run(c))

    print("666")