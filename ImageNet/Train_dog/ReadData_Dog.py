import tensorflow as tf
import numpy as np
from PIL import Image

sess = tf.InteractiveSession()


dog_x = np.fromfile('dog_all.dat', dtype='uint8')
dog_y = np.fromfile('dog_label_all.dat', dtype='float32')
dog_x = dog_x.astype('float32')
dog_x = np.reshape(dog_x, [-1, 400, 400, 3])
dog_y = np.reshape(dog_y, [-1, 2])
print(dog_x.shape)
print(dog_y.shape)

x_train = dog_x[0:600]
y_train = dog_y[0:600]
x_test = dog_x[600:]
y_test = dog_y[600:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x = tf.placeholder(tf.float32, [None, 400, 400, 3])
y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.get_variable('W1', [5, 5, 3, 10], initializer=tf.contrib.layers.xavier_initializer())

L1 = tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

'''
Tensor("Conv2D:0", shape=(?, 200, 200, 10), dtype=float32)
Tensor("Relu:0", shape=(?, 200, 200, 10), dtype=float32)
Tensor("MaxPool:0", shape=(?, 200, 200, 10), dtype=float32)
'''

W2 = tf.get_variable('W2', [5, 5, 10, 20], initializer=tf.contrib.layers.xavier_initializer())

L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
L2_vec = tf.reshape(L2, [-1, 100*100*2])

'''
Tensor("Conv2D_1:0", shape=(?, 100, 100, 20), dtype=float32)
Tensor("Relu_1:0", shape=(?, 100, 100, 20), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 100, 100, 20), dtype=float32)
Tensor("Reshape:0", shape=(?, 20000), dtype=float32)
'''
W3 = tf.get_variable('W3', [100*100*2, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
h = tf.matmul(L2_vec, W3) + b


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs = 2

print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(0, 5):
        i = i*100
        feed_dict = {x: x_train[i:i+99], y: y_train[i:i+99]}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/6
    print('Epoch:', '%04d' % (epoch + 1), 'Cost:', '{:.9f}'.format(avg_cost))

print('Learning Finished')


correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      x: x_test, y: y_test}))


'''
img = W1.eval()
print(img)
img_a = np.swapaxes(img, 0, 3)
print(img_a.shape)

a = []
for i in range(0, 3):
    
    b = Image.fromarray(img[i])

'''
