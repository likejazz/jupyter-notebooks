import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, name="X")
x_data = np.array([[1,1,4,5,6,8,9]], dtype=np.float)
x_data = x_data.reshape(-1, 1)

t = tf.placeholder(tf.float32, name="t")
t_data = np.array([[107],[83],[70]], dtype=np.float)

W = tf.Variable([[1,5,3,7,4,5,66], [2,4,23,1,2,1,1], [4,4,1,1,1,1,2]], dtype=tf.float32)
y = tf.matmul(W, X)

cost = tf.reduce_sum(1/2 * tf.square(y - t))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(50):
        cost_val = sess.run(cost, feed_dict={X: x_data, t: t_data})
        sess.run(optimizer, feed_dict={X: x_data, t: t_data})
        print(step, cost_val)

    print(sess.run(W))
    print(sess.run(y, feed_dict={X: x_data}))
