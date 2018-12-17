# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

def init():
    pass

def sample_Z(size=[128,100]):
    return np.random.uniform(-1, size=size)
#随机生成种子Z

def plot(samples, grid=(3,3), size=(28,28), name='tmp'):
    for i, sample in enumerate(samples):
        ax = plt.subplot('%d%d%d' %(grid[0], grid[1], i))
        plt.axis('off')
        plt.imshow(sample.reshape(size), cmap=plt.cm.gray)
    plt.savefig('%s.png' %name)
#绘制样本

#定义生成对抗网络结构

def init_xavier(size):
    dim_in = size[0]
    dim_out = size[1]
    var = 2. / (dim_in + dim_out)
    stddev = tf.sqrt(var)
    return tf.random_normal(shape=size, stddev=stddev)
#xavier初始化矩阵

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
#输入样本

with tf.name_scope('D'):
    D_W1 = tf.Variable(init_xavier([784, 128]), name='W1')
    D_b1 = tf.Variable(tf.zeros([128]), name='b1')
    D_W2 = tf.Variable(init_xavier([128, 1]), name='W2')
    D_b2 = tf.Variable(tf.zeros([1]), name='b2')
#鉴别器参数

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
#生成种子

with tf.name_scope('G'):
    G_W1 = tf.Variable(init_xavier([100, 128]), name='W1')
    G_b1 = tf.Variable(tf.zeros([128]), name='b1')
    G_W2 = tf.Variable(init_xavier([128, 784]), name='W2')
    G_b2 = tf.Variable(tf.zeros([784]), name='b2')
#生成器参数

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
#鉴别器_对样本做出真假鉴定

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_out = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_out)
    return G_prob
#生成器_由种子生成样本

G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)
with tf.name_scope('loss'):
    D_loss = - tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake), name='D')
    G_loss = - tf.reduce_mean(tf.log(D_fake), name='G')
#损失函数
with tf.name_scope('solve'):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=[D_W1, D_b1, D_W2, D_b2], name='D')
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=[G_W1, G_b1, G_W2, G_b2], name='G')
#优化方法

batch_size = 128
batch_num = 20000
interval = 1000
mnist = input_data.read_data_sets('../../Data/mnist/', one_hot=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(batch_num):
        X_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(D_solver, feed_dict={X: X_batch, Z: sample_Z([batch_size, 100])})
        sess.run(G_solver, feed_dict={Z: sample_Z([batch_size, 100])})
        if i % interval == 0:
            D_loss_val, G_loss_val = sess.run([D_loss, G_loss], feed_dict={X: X_batch, Z: sample_Z([batch_size, 100])})
            print('%d:\n\tD_loss: %.6f\n\tG_loss: %.6f\n' %(i, D_loss_val, G_loss_val))
            samples = sess.run(G_sample, feed_dict={Z: sample_Z([9, 100])})
            plot(samples, name='tmp')
        #输出损失,并绘制效果
#训练