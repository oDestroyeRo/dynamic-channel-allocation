import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# x = tf.constant([[1., 1., 1.], [2., 2., 1.], [3., 2., 1.], [3., 2., 1.]])
x = tf.Variable([[1., 1., 1.]])
x1 = tf.Variable([[1., 1., 1.]])
x2 = tf.Variable([[1., 1., 1.]])

re = tf.Variable([x , x1 , x2])


with tf.Session() as sess:
    #  print(product.eval()) 
    # x4 = x+ x1 + x2
    # x4 = x4/3
    # print(x4.eval()) 

    # print(tf.reduce_mean(x).eval()) # 1.5
    print(tf.reduce_mean(re).eval()) # [1.5, 1.5]
    # print(tf.reduce_mean(x, 1).eval())  # [1.,  2.]

