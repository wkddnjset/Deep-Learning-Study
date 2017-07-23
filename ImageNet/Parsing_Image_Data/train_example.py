import csv
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
# tofile - fromfile로 data 불러오기
x = np.fromfile('cat_all.dat', dtype='uint8')
y = np.fromfile('cat_label_all.dat', dtype='float32')
x = x.astype('float32')
print(x.shape)
print(y.shape)
x = np.reshape(x, [-1, 400, 400, 3])
y = np.reshape(y, [-1, 2])
print(x.shape)
x_train = x[0:700]
y_train = y[0:700]
x_test = x[701:]
y_test = y[701:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

X = tf.placeholder(tf.float32, [None, 400, 400, 3])
Y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 400, 400, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 400, 400, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 200, 200, 32), dtype=float32)
'''
W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_vec = tf.reshape(L2, [-1, 100*100*64])
'''
Tensor("Conv2D_1:0", shape=(?, 200, 200, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 200, 200, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 100, 100, 64), dtype=float32)
Tensor("Reshape:0", shape=(?, 640000), dtype=float32)
'''
W3 = tf.get_variable("W3", shape=[100*100*64, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L2_vec, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs = 2

print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(0, 7):
        i = i*100
        feed_dict = {X: x_train[i:i+99], Y: y_train[i:i+99]}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/4
    print('Epoch:', '%04d' % (epoch + 1), 'Cost:', '{:.9f}'.format(avg_cost))

print('Learning Finished')


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: x_test, Y: y_test}))
'''
W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_vec = tf.reshape(L2, [-1, 100*100*64])

W3 = tf.Variable("W3", shape=[100*100*64, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L2_vec, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs = 15
batch_size = 100

print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    feed_dict = {X:x_train, Y:y_train}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c
    print('Epoch:','%04d' %(epoch +1),'Cost:','{:.9f}'.format(avg_cost))
    print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: x_test, Y: y_test}))

# Get one and predict
r = random.randint(0, len(x_test) - 1)
print("Label: ", sess.run(tf.argmax(y_test[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: x_test[r:r + 1]}))
'''