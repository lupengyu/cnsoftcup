#coding=utf-8
import tensorflow as tf
import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
#import csv
start_time = time.time()
in_file = 'sales_sample_20170310.csv'
print("Reading the data from", in_file)
full_data = pd.read_csv(in_file)
day_id_data = np.array(full_data['day_id'])
sale_nbr_data = np.array(full_data['sale_nbr'])
buy_nbr_data = np.array(full_data['buy_nbr'])
cnt_data = np.array(full_data['cnt'])
round_data = np.array(full_data['round'])
data_cnt = len(full_data)
cnt_matrix = np.zeros((91, 541556), dtype = np.float32)
round_matrix = np.zeros((91, 541556), dtype = np.float32)
relationship_matrix = np.zeros((7491, 7491), dtype = np.int32)
parent_sale = []
parent_buy = []

print("Build relationship")
cursor = 1
for i in range(data_cnt):
    day_item = day_id_data[i] - 1
    sale_item = str(sale_nbr_data[i])
    buy_item = str(buy_nbr_data[i])
    
    if sale_item[0] == 'O':
        sale_cursor = int(sale_item[1:]) + 42
    elif sale_item[0] == 'C':
        sale_cursor = int(sale_item[1:]) - 1
    elif sale_item[0] == 'P':
        sale_cursor = 7490
    #缺省数据忽略
    else:
        continue
    
    if buy_item[0] == 'O':
        buy_cursor = int(buy_item[1:]) + 42
    elif buy_item[0] == 'C':
        buy_cursor = int(buy_item[1:]) - 1
    elif buy_item[0] == 'P':
        buy_cursor = 7490
    #缺省数据忽略
    else:
        continue
    
    #build relationship
    if relationship_matrix[sale_cursor][buy_cursor] > 0 or relationship_matrix[buy_cursor][sale_cursor] > 0:
        #add to matrix
        cnt_matrix[day_item][relationship_matrix[sale_cursor][buy_cursor]] = int(cnt_data[i])
        round_matrix[day_item][relationship_matrix[sale_cursor][buy_cursor]] = int(round_data[i])
    else :
        relationship_matrix[sale_cursor][buy_cursor] = cursor
        relationship_matrix[buy_cursor][sale_cursor] = cursor
        parent_sale.append(sale_item)
        parent_buy.append(buy_item)
        #add to matrix
        cnt_matrix[day_item][cursor] = float(cnt_data[i])
        round_matrix[day_item][cursor] = float(round_data[i])
        cursor += 1
print("Prediction start")
max_steps = 10000
round_x = np.zeros((91, 24, 24, 3), dtype = np.float32)
round_y = np.zeros((91, 1728), dtype = np.float32)
for i in range(37):
    for j in range(24):
        for k in range(24):
            for m in range(3):
                round_x[i][j][k][m] = round_matrix[i][72*j+3*k+m+1]
                round_y[i][72*j+3*k+m] = round_matrix[i][72*j+3*k+m+1]
train_round_x = round_x[:36]
train_round_y = round_y[1:37]
print(len(train_round_x))
print(len(train_round_y))

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

image_holder = tf.placeholder(tf.float32, [None, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [None, 1728])

#first network - CNN
weight1 = variable_with_weight_loss(shape = [5, 5, 3, 64], stddev = 5e-2, 
                                    w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding = 'SAME')
bias1 = tf.Variable(tf.constant(0.0, shape = [64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
                       padding = 'SAME')
norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

#second network - CNN
weight2 = variable_with_weight_loss(shape = [5, 5, 64, 128], stddev = 5e-2, 
                                    w1 = 0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding = 'SAME')
bias2 = tf.Variable(tf.constant(0.1, shape = [128]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
                       padding = 'SAME')

#third network 4608 = 36*128  [90,1728] vs. [36,1728]
#128*36 [90,1728] vs. [128,1728]
reshape = tf.reshape(pool2, [36, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape = [128, 384], stddev = 0.04, w1 = 0.004)
bias3 = tf.Variable(tf.constant(0.1, shape = [384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

keep_prob = tf.placeholder(tf.float32)
local3_drop = tf.nn.dropout(local3, keep_prob)

#4th network
weight4 = variable_with_weight_loss(shape = [384, 192], stddev = 0.04, w1 = 0.004)
bias4 = tf.Variable(tf.constant(0.1,shape = [192]))
local4 = tf.nn.relu(tf.matmul(local3_drop, weight4) + bias4)

local4_drop = tf.nn.dropout(local4, keep_prob)

#5th network
weight5 = variable_with_weight_loss(shape = [192, 1728], stddev = 1 / 192.0, w1 = 0.0)
bias5 = tf.Variable(tf.constant(0.0, shape = [1728]))
logits = tf.add(tf.matmul(local4_drop, weight5), bias5)

def loss(logits, labels):
    labels = tf.cast(labels, tf.float32)
    cross_entropy_mean = tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits,1e-10,1.0)),reduction_indices=[1]))
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

loss = loss(logits, label_holder)
#train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss],
                             feed_dict = {image_holder: train_round_x, 
                                          label_holder: train_round_y,
                                          keep_prob: 1.0})
    duration = time.time() - start_time
    if step % 200 == 0 :
        sec_per_batch = float(duration)
        format_str = ('step %d, loss = %.5f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, sec_per_batch))    