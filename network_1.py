#coding=utf-8
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import random
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
train_cnt = np.zeros((91, 1000), dtype = np.float32)
train_round = np.zeros((91, 1000), dtype = np.float32)
item_round_x = np.zeros((1, 1000), dtype = np.float32)
item_round_y = np.zeros((1, 1000), dtype = np.float32)
for i in range(91):
    for j in range(1000):
        cursor = j + 1
        train_cnt[i][j] = cnt_matrix[i][cursor]
        train_round[i][j] = round_matrix[i][cursor]
train_cnt_x = train_cnt[:90]
train_cnt_y = train_cnt[1:]
train_round_x = train_round[:90]
train_round_y = train_round[1:]
print(len(train_round_x))
print(len(train_round_x[0]))
print(len(train_round_y))
print(len(train_round_y[0]))
for i in range(1000):
    if int(train_round_x[89][i]) != 0:
        print(i)

xs = tf.placeholder(tf.float32, [1, 1000]) # 28x28
ys = tf.placeholder(tf.float32, [1, 1000])
#size = tf.placeholder(tf.int32)

Weights_1 = tf.Variable(tf.random_normal([1000, 1000]))
biases_1 = tf.Variable(tf.random_normal([1000]))
Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1
'''
Weights_2 = tf.Variable(tf.random_normal([4000, 8000]))
biases_2 = tf.Variable(tf.random_normal([8000]))
Wx_plus_b_2 = tf.matmul(Wx_plus_b_1, Weights_2) + biases_2

Weights_3 = tf.Variable(tf.random_normal([8000, 1000]))
biases_3 = tf.Variable(tf.random_normal([1000]))
Wx_plus_b_3 = tf.matmul(Wx_plus_b_2, Weights_3) + biases_3
'''
#tf.clip_by_value(Wx_plus_b,1e-10,1.0)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(Wx_plus_b_1,1e-10,1.0)),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        j = random.randint(0, 89)
        item_round_x[0] = train_round_x[j]
        item_round_y[0] = train_round_y[j]
        _, loss = sess.run([train_step, cross_entropy], feed_dict={xs: item_round_x, ys: item_round_y})
        if i % 50 == 0:
            print(i, loss)
    normalized_data = []
    predict = []
    for i in range(91):
        normalized_data.append(round_matrix[i][3])
    w_1, b_1 = sess.run([Weights_1, biases_1])
    #w_2, b_2 = sess.run([Weights_2, biases_2])
    #w_3, b_3 = sess.run([Weights_3, biases_3])
    item_round_x[0] = train_round_y[-1]
    print(item_round_x)
    print("-1", int(item_round_x[0][2]))
    for i in range(30):
        result = sess.run(Wx_plus_b_1, feed_dict={xs: item_round_x})
        #result_2 = np.dot(result_1 , w_2) + b_2
        #result = np.dot(result_2 , w_3) + b_3
        predict.append(int(result[0][2]))
        print(i, int(result[0][2]))
        item_round_x = result
    '''
    for i in range(30):
            next_seq = sess.run(Wx_plus_b, feed_dict={xs: item_round_x})
            predict.append(int(next_seq[0][49]))
            print(i, int(next_seq[0][49]))
            print(i, int(next_seq[0][50]))
            prev_seq = next_seq
    '''
    plt.figure()
    plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
    plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
    plt.show()