#coding=utf-8
import pandas as pd
import numpy as np
import time
import tensorflow as tf
#import matplotlib.pyplot as plt
#import random
#import csv
start_time = time.time()
in_file = 'sales_sample_20170310.csv'
print("Reading the data from", in_file)
#full_data size:857922824
full_data = pd.read_csv(in_file)
day_id_data = np.array(full_data['day_id'])
sale_nbr_data = np.array(full_data['sale_nbr'])
buy_nbr_data = np.array(full_data['buy_nbr'])
#max = 110540
cnt_data = np.array(full_data['cnt'])
#max = 93000000
round_data = np.array(full_data['round'])
data_cnt = len(full_data)

'''
C      43个          No:0 - 42
O      7447个     No:43 - 7489
PAX    1个             No:7490
总计          7491个
关系数量   541555个  = 35 * 15473
'''
cnt_matrix = np.zeros((91, 541556), dtype = np.float32)
round_matrix = np.zeros((91, 541556), dtype = np.float32)
relationship_matrix = np.zeros((7491, 7491), dtype = np.int32)
parent_sale = []
parent_buy = []

print("Build relationship")
#build relationship Time:20s
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
seq_size = 3
batch_size = 1000
train_x = np.zeros((87, seq_size, batch_size), dtype = np.float32)
train_y = np.zeros((87, batch_size), dtype = np.float32)
for i in range(87):
    for j in range(batch_size):
        train_y[i][j] = round_matrix[i + 3][j + 1]
for i in range(87):
    for j in range(seq_size):
        for k in range(batch_size):
            train_x[i][0][k] = round_matrix[i][k + 1]
            train_x[i][1][k] = round_matrix[i + 1][k + 1]
            train_x[i][2][k] = round_matrix[i + 2][k + 1]
train_x_change = np.reshape(train_x, [87, 3000])
X = tf.placeholder(tf.float32, [None, 3000])
Y = tf.placeholder(tf.float32, [None, 1000])

Weights_layout_1 = tf.Variable(tf.random_normal([3000, 1000], -1.0, 1.0))
Basic_layout_1 = tf.Variable(tf.zeros([1, 1000]) + 0.1)
Wx_plus_layout_1 = tf.matmul(X, Weights_layout_1) + Basic_layout_1
Wx_plus_layout_1_relu = tf.nn.relu(Wx_plus_layout_1)

#tf.log(tf.clip_by_value(X_layout_1,1e-10,1.0))
#tf.log(X_layout_1)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(Wx_plus_layout_1_relu,1e-10,1.0)), 
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, loss, w, result, result_relu = sess.run([train_step, cross_entropy, Weights_layout_1,
                                                    Wx_plus_layout_1, Wx_plus_layout_1_relu], feed_dict={X: train_x_change, Y: train_y})
        if i % 50 == 0:
            print(i, loss, w[0][0])
            print("result", result)
            print("result_relu", result_relu)         
    print("-------------------")
    w, b = sess.run([Weights_layout_1, Basic_layout_1])
    print(w, b)
