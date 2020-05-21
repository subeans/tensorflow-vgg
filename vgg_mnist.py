%load_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import utils
import vgg16 as vgg19
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
batch_size = 300

images = tf.placeholder(tf.float32, [None, 28, 28, 1])
true_out = tf.placeholder(tf.float32, [None, 10])
train_mode = tf.placeholder(tf.bool)

vgg = vgg26.Vgg16()
vgg.build(images, train_mode)

print(vgg.get_var_count())

# test classification
sess.run(tf.global_variables_initializer())

batch = mnist.train.next_batch(batch_size)
batch_img = batch[0].reshape((-1,28,28,1))
batch_lbl = batch[1]

print(batch_img.shape, batch_lbl.shape)

print (np.argmax(batch_lbl[0]))
print (np.argmax(batch_lbl[1]))
print (np.argmax(batch_lbl[2]))
print (np.argmax(batch_lbl[3]))


plt.figure()
plt.imshow(batch_img[0,:,:,0])
plt.figure()
plt.imshow(batch_img[1,:,:,0])
plt.figure()
plt.imshow(batch_img[2,:,:,0])
plt.figure()
plt.imshow(batch_img[3,:,:,0])

cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(true_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

vbatch = mnist.validation.next_batch(500)
vbatch_img = vbatch[0].reshape((-1,28,28,1))
vbatch_lbl = vbatch[1]

print ('accuracy: ', sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))

for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    batch_img = batch[0].reshape((-1,28,28,1))
    batch_lbl = batch[1]
    sess.run(train, feed_dict={images: batch_img, true_out: batch_lbl, train_mode: True})
    if i % 50 == 0:
        print( 'iteration: ', i)
        vbatch = mnist.validation.next_batch(500)
        vbatch_img = vbatch[0].reshape((-1,28,28,1))
        vbatch_lbl = vbatch[1]
        print ('accuracy: ', sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))
vbatch = mnist.validation.next_batch(2000)
vbatch_img = vbatch[0].reshape((-1,28,28,1))
vbatch_lbl = vbatch[1]
print(sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))
