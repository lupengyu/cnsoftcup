import numpy as np
import tensorflow as tf
train_cnt_item = np.zeros((91, 10), dtype = np.float32)
train_round_item = np.zeros((91, 10), dtype = np.float32)
train_cnt_x = np.zeros((90, 3, 10, 1), dtype = np.float32)
train_cnt_y = np.zeros((90, 3, 10), dtype = np.float32)
'''
for j in range(len(train_cnt_item) - 1):
    train_cnt_x.append(np.expand_dims(train_cnt_item[j], axis=2).tolist())
    train_cnt_y.append(train_cnt_item[j + 1].tolist())
print(train_cnt_x)
print(len(train_cnt_x))
print(len(train_cnt_x[0]))
print(len(train_cnt_x[0][0]))
print(train_cnt_y)
print(len(train_cnt_y))
print(len(train_cnt_y[0]))
'''
'''
Tensor("Placeholder:0", shape=(?, 3, 1, 10), dtype=float32)
Tensor("Placeholder_1:0", shape=(?, 3, 10), dtype=float32)
Tensor("W/read:0", shape=(60, 10), dtype=float32)
Tensor("b/read:0", shape=(10,), dtype=float32)
<tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl.BasicLSTMCell object at 0x0000025C9B0460B8>
'''
seq_size = 3
input_dim = 1
hidden_layer_size = 6
X = tf.placeholder(tf.float32, [None, 3, 10, 1])
Y = tf.placeholder(tf.float32, [None, 3, 10])
W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
b = tf.Variable(tf.random_normal([1]), name='b')
cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
out = tf.matmul(outputs, W_repeated) + b
out = tf.squeeze(out)
loss = tf.reduce_mean(tf.square(out - Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

loss = tf.reduce_mean(tf.square(out - Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
with tf.Session() as sess:
    #tf.get_variable_scope().reuse_variables()
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        _, loss_ = sess.run([train_op, loss], feed_dict={X: train_cnt_x, Y: train_cnt_y})
        if step % 10 == 0:
            print(step, loss_)
    prev_seq = train_cnt_x[-1]
    predict = []
    for i in range(12):
        next_seq = sess.run(out, feed_dict={X: [prev_seq]})
        predict.append(next_seq[-1])
        prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
    print(predict)