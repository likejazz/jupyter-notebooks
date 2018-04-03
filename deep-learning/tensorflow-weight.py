import tensorflow as tf
import numpy as np

X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(np.random.random(), name="weight")
pred = X * W
cost = tf.reduce_sum(tf.pow(pred - Y, 2))
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(200):
        x = np.array(np.random.random()).reshape((1, 1, 1, 1))
        y = x * 3
        (_, c) = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
        # print (c)
    print(W.eval())
