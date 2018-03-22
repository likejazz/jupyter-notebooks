import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# for d in ['/device:GPU:0', '/device:GPU:1']:
for d in ['/cpu:0']:
    with tf.device(d):
        with tf.name_scope('layer1'):
            W1 = tf.Variable(tf.random_normal([784, 4096], stddev=0.01))
            net1 = tf.matmul(X, W1)
            out1 = tf.nn.relu(net1)

        with tf.name_scope('layer2'):
            W2 = tf.Variable(tf.random_normal([4096, 256], stddev=0.01))
            out2 = tf.nn.relu(tf.matmul(out1, W2))

        with tf.name_scope('output'):
            W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
            out3 = tf.matmul(out2, W3)

        with tf.name_scope('optimizer'):
            cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out3, labels=Y))
            # optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)
            optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost, global_step=global_step)
            # optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cost, global_step=global_step)

# --

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    start = time.time()

    for epoch in range(5000):
        total_cost = 0

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

        if epoch % 100 == 0:
            is_correct = tf.equal(tf.argmax(out3, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

            print('Epoch: %4d, loss = %3.3f,  \tacc = %3.3f' % (
                (epoch + 1),
                total_cost,
                acc
            ))

    end = time.time()

    print('Completed!')

    is_correct = tf.equal(tf.argmax(out3, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('Accuracy:', sess.run(accuracy,
                                feed_dict={X: mnist.test.images,
                                           Y: mnist.test.labels}))
    print('Elapsed(train): %12.2f' % (end - start))
