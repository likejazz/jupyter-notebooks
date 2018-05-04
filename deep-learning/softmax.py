# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy_data = np.array([
    [1, 2, 1, 1, 0, 0, 1],
    [1, 3, 2, 2, 0, 0, 1],
    [1, 3, 4, 3, 0, 0, 1],
    [1, 5, 5, 2, 0, 1, 0],
    [1, 7, 5, 1, 0, 1, 0],
    [1, 2, 5, 2, 0, 1, 0],
    [1, 6, 6, 4, 1, 0, 0],
    [4, 7, 7, 3, 1, 0, 0],
])

x_data = xy_data[:, :4]
y_data = xy_data[:, 4:]

# row : infinity, col : 3 for x0/x1/x2/x3
X = tf.placeholder("float", [None, 4])
# row : infinity, col : 3 for A/B/C target class which is encoded in one-hot representation
Y = tf.placeholder("float", [None, 3])

# output layer

# row : 4 dimensions for x, col : 3 dimensions for y
W = tf.Variable(tf.zeros([4, 3]))
# softmax, (None x 4) * ( 4 x 3 )
y = tf.nn.softmax(tf.matmul(X, W))

# %%
# training
# cross entropy cost
# Y = one-hot vector
# y = softmaxed value
cost = - tf.reduce_mean(tf.reduce_sum(Y * tf.log(y) + (1 - Y) * tf.log(1 - y), axis=1))  # binomial cross-entropy error
# cost = - tf.reduce_mean(tf.reduce_sum(Y * tf.log(y), axis=1)) # multinomial cross-entropy error
# cost = 1 - tf.reduce_mean(tf.reduce_sum(Y * y, axis=1))       # strange cross-entropy variation I have created.
train = tf.train.GradientDescentOptimizer(tf.Variable(0.001)).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %%
results = []
for i in range(20000 + 1):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if i % 20 == 0:
        result = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print(i, result)
        results.append(result)

# %%
# plot loss changes
plt.plot(results)
plt.show()

# %%
# inference
np.set_printoptions(suppress=True)

p = sess.run(y, feed_dict={X: [[1, 2, 1, 1]]})  # 0 0 1 -> 2
print(p, sess.run(tf.argmax(p, 1)))

p = sess.run(y, feed_dict={X: [[1, 5, 5, 2]]})  # 0 1 0 -> 1
print(p, sess.run(tf.argmax(p, 1)))

p = sess.run(y, feed_dict={X: [[1, 2, 1, 1], [1, 5, 5, 2]]})  # 2, 1
print(p, sess.run(tf.argmax(p, 1)))

p = sess.run(y, feed_dict={X: [[4, 7, 7, 3]]})  # 0
print(p, sess.run(tf.argmax(p, 1)))

p = sess.run(y, feed_dict={X: [[1, 7, 7, 3]]})  # 0?
print(p, sess.run(tf.argmax(p, 1)))
