import numpy as np
import tensorflow as tf
from PIL import Image
import random

cat_x = np.fromfile('../Train_cat/cat_all.dat', dtype='uint8')
cat_x = cat_x.astype('float32')
cat_y = np.fromfile('../Train_cat/cat_label_all.dat', dtype='float32')
dog_x = np.fromfile('../Train_dog/dog_all.dat', dtype='uint8')
dog_x = dog_x.astype('float32')
dog_y = np.fromfile('../Train_dog/dog_label_all.dat', dtype='float32')

cat_x = np.reshape(cat_x, [-1, 400, 400, 3])
cat_y = np.reshape(cat_y, [-1, 2])
dog_x = np.reshape(dog_x, [-1, 400, 400, 3])
dog_y = np.reshape(dog_y, [-1, 2])

all_data = []
for i in range(0, 1500):
    a = i/2
    b = a = round((i+0.01)/2)-1
    if i % 2 == 0:
        all_data.append([[cat_y[a]], [cat_x[a]]])
    else:
        all_data.append([[dog_y[b]], [dog_x[b]]])
'''
print(cat_x.shape)
print(cat_x.dtype)
print(cat_y.shape)
print(cat_y.dtype)
print(dog_x.shape)
print(dog_x.dtype)
print(dog_y.shape)
print(dog_y.dtype)
'''
train_label = []
for i in range(0, 1000):
    train_label.append(all_data[i][0])
train_data = []
for i in range(0, 1000):
    train_data.append(all_data[i][1])
test_label = []
for i in range(1000, 1500):
    test_label.append(all_data[i][0])
test_data = []
for i in range(1000, 1500):
    test_data.append(all_data[i][1])

train_label = np.reshape(train_label, [-1, 2])
train_data = np.reshape(train_data, [-1, 400, 400, 3])
test_label = np.reshape(test_label, [-1, 2])
test_data = np.reshape(test_data, [-1, 400, 400, 3])
print(train_label.shape)
print(train_data.shape)
print(test_label.shape)
print(test_data.shape)


X = tf.placeholder(tf.float32, [None, 400, 400, 3])
Y = tf.placeholder(tf.float32, [None, 2])


W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_vec = tf.reshape(L2, [-1, 100*100*64])

W3 = tf.get_variable("W3", shape=[100*100*64, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L2_vec, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs = 20

print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(0, 10):
        i = i*100
        feed_dict = {X: train_data[i:i+99], Y: train_label[i:i+99]}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/10
    print('Epoch:', '%04d' % (epoch + 1), 'Cost:', '{:.9f}'.format(avg_cost))

print('Learning Finished')

#강아지 = 1 고양이 = 0
prediction = tf.argmax(hypothesis, 1)
target = tf.argmax(Y, 1)
correct_prediction = tf.equal(prediction, target)
#correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
print(correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
print('Accuracy: %.2df' % sess.run(accuracy * 100, feed_dict={X: test_data, Y: test_label}))
