import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# set up the layers
hidden1 = 600
hidden2 = 600
hidden3 = 600
output = 10
batchSize = 100
imSize = 28 * 28

# set up input type and output type
x = tf.placeholder(tf.float32, shape = [None, imSize])
y = tf.placeholder(tf.float32)

# set up the nn
def nN(input):
    # set up connection
    input_l1 = {'weights': tf.Variable(tf.random_normal([imSize, hidden1], mean=0.0, stddev=0.1)),
                      'biases': tf.Variable(tf.random_normal([hidden1], mean=0.0, stddev=0.1))}

    l1_l2 = {'weights': tf.Variable(tf.random_normal([hidden1, hidden2], mean=0.0, stddev=0.1)),
                      'biases': tf.Variable(tf.random_normal([hidden2], mean=0.0, stddev=0.1))}

    l2_l3 = {'weights': tf.Variable(tf.random_normal([hidden2, hidden3], mean=0.0, stddev=0.1)),
                      'biases': tf.Variable(tf.random_normal([hidden3], mean=0.0, stddev=0.1))}

    l3_output = {'weights': tf.Variable(tf.random_normal([hidden3, output], mean=0.0, stddev=0.1)),
                    'biases': tf.Variable(tf.random_normal([output], mean=0.0, stddev=0.1))}

    # set up layers
    l1 = tf.add(tf.matmul(input, input_l1['weights']), input_l1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, l1_l2['weights']), l1_l2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, l2_l3['weights']), l2_l3['biases'])
    l3 = tf.nn.relu(l3)
    return tf.add(tf.matmul(l3, l3_output['weights']), l3_output['biases'])


def train(x):
    prediction = nN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    n_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples / batchSize)):
                mnist_x, mnist_y = mnist.train.next_batch(batchSize)
                i, c = sess.run([optimizer, cost], feed_dict={x: mnist_x, y: mnist_y})
                epoch_loss += c
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            print('Epoch', epoch, 'completed out of', n_epochs - 1, 'accuracy:', tf.reduce_mean(tf.cast(correct, tf.float32)).eval({x: mnist.test.images, y: mnist.test.labels}))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train(x)
