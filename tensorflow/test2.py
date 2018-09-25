#! /usr/bin/python3.6
import tensorflow as tf
import numpy as np
import os

# clear up terminal - lots of output coming
os.system('clear')
lineStr = '\n---------------------------'
print(f'TENSOR TESTING:{lineStr}')


print('we are going to solve the following:')
print(f'b = ??\nc = 1\nd = b + c\ne = c + 2\na = d * 2\nusing tensorflow{lineStr}')


# create tensorflow constant
const = tf.constant(2.0, name='const')

# create our mystery variable
# (objectType,  [cage, dimensional-array num], name)
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

print(f'Making Variables:\nSo i just created:\n{b}\nand\n{c}\nwhich are tf variables. Their first arg, a float, is their initial value')
print(f'I am also giving them an optional name= argument - this will become useful later for visualization')
print(f'Note that these variables are not created yet, just tf objects that are preparing to create them{lineStr}')

# create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')


print(f'Tensorflow has tons of operations, I just made two adds and a multiply')
print(f'I just made\n{d}\n{e}\n{a}\nwhich are three operations ready to go{lineStr}')

# setup the variable initialization
init_op = tf.global_variables_initializer()

print(f'I have just created another operation, the initializer, which will create all of my globals (variables I just set up{lineStr}')


# start the session
with tf.Session() as sess:

    # initialize the varaibles
    sess.run(init_op)

    # compute the output of the graph
    a_out = sess.run(a, feed_dict = {b: np.arange(0,10)[:, np.newaxis]})
    print(f'variable a is {a_out}')
    c_out = sess.run(c, feed_dict = {b: np.arange(0,10)[:, np.newaxis]})
    print(f'variable c is {c_out}')
    e_out = sess.run(e, feed_dict = {b: np.arange(0,10)[:, np.newaxis]})
    print(f'variable e is {e_out}')
    d_out = sess.run(d, feed_dict = {b: np.arange(0,10)[:, np.newaxis]})
    print(f'variable d is {d_out}')


    # types
    print(f'a_out is type {type(a_out)}')
    for num in range(0,10):
        print(f'{lineStr}\nWhen b = {num}\na = {a_out[num]}\nd = {d_out[num]}')




print('\n\n')
