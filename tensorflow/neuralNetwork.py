#! /usr/bin/python3.6
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100


# declare the training data placeholders
print('Setting up "x" as pixel placeholder')
print('the test data is 28 x 28 pixels, so 784 pixels')
x = tf.placeholder(tf.float32, [None, 784])
print('Setting up "y" as output data placeholder')
print('10 digits')
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
print('next im going to set up the weights and bias for the hidden layer')
print('the hidden layer does smaller computers to try and answer the bigger question')
print('if youre looking for a bus, it might be easier to say...\n"Are there wheels?"\n"Is it box-shaped?"\n"Is it bigger than a card?"\n')
print('we have 300 nodes in hidden layer and 784 pixels...\nso the size of the weight tensor is [784,300]')

W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')


print('now we are making the weight and bias again, this time for the output layer')
print('our output layer size of weight tensor should be [300,10]')
print('300 from the above number of nodes in hidden layer and 10 from the number of outputs (digits) since we are going to try and detect text numebrs')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')


print('now we need to setup node inputs and activation functions of hidden layer nodes')
print('the first line will do MATrix MULtiplication (matmul) by the input vector x')
print('the arg of our hidden-layer bias, b1, is included')
print('finally - we finalize the hidden_out operation by adding a REctified Linear Unit (relu) activation function to the matrix multiplication plus bias')
print('tensorflow has a RELU activation ready in tf.nn.relu')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)


print('now that weve set up inputs and activated functions of the hidden layer we need to get the output layer ready')
print('we need to multiply the output of the hidden layer "hidden_out" by our output weight layer "W2" and include the output bias "B2"')
print('we will use softmax activation for the output layer')
print('tensorflow provides this with tf.nn.softmax')


# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))


print('the first line clips y so that we are as close between 0 and 1 as-possible with never having a log(0) situation or a log(1) situation')



y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))


print('now we will add an optimiser with our globals above')


# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


print('now lets initialize all of these crazy variables')


# finally setup the initialisation operator
init_op = tf.global_variables_initializer()


print('lets define what correct means and how to check accuracy')


# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the training session
with tf.Session() as sess:
    # init the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                    feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels}))
