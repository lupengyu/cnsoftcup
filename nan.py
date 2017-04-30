#coding=utf-8
import pandas as pd
import numpy as np
import time
#import tensorflow as tf
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
#1 - 541555
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
        parent_sale.append(sale_item)
        parent_buy.append(buy_item)
        #add to matrix
        cnt_matrix[day_item][cursor] = float(cnt_data[i])
        round_matrix[day_item][cursor] = float(round_data[i])
        cursor += 1
print("Prediction start")
#15473 * 35
seq_size = 3
input_dim = 1
train_cnt_item = np.zeros((91, 15473), dtype = np.float32)
train_round_item = np.zeros((91, 15473), dtype = np.float32)
for j in range(91):
    cursor = 0
    for k in range(15473):
        train_cnt_item[j][cursor] = cnt_matrix[j][cursor + 1]
        train_round_item[j][cursor] = round_matrix[j][cursor + 1]
        cursor += 1
train_cnt_x, train_cnt_y = [], []
train_round_x, train_round_y = [], []
for j in range(len(train_cnt_item) - seq_size - 1):
    train_cnt_x.append(np.expand_dims(train_cnt_item[j : j + seq_size], axis=1).tolist())
    train_cnt_y.append(train_cnt_item[j + 1 : j + seq_size + 1].tolist())
    train_round_x.append(np.expand_dims(train_round_item[j : j + seq_size], axis=1).tolist())
    train_round_y.append(train_round_item[j + 1 : j + seq_size + 1].tolist())
print(train_round_x)
print(train_round_y)
#541555 * 1 Time:83h
'''
hidden_layer_size = 6
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])
W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
b = tf.Variable(tf.random_normal([1]), name='b')
cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
out = tf.matmul(outputs, W_repeated) + b
out = tf.squeeze(out)
loss = tf.reduce_mean(tf.square(out - Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

prediction_cnt = np.zeros((91, 541556), dtype = np.float32)
prediction_round = np.zeros((91, 541556), dtype = np.float32)
for i in range(541555):
    cursor = i + 1
    train_cnt_item = []
    train_round_item = []
    for j in range(91):
        train_cnt_item.append(cnt_matrix[j][cursor])
        train_round_item.append(round_matrix[j][cursor])
    train_cnt_x, train_cnt_y = [], []
    train_round_x, train_round_y = [], []
    for j in range(len(train_cnt_item) - seq_size - 1):
        train_cnt_x.append(np.expand_dims(train_cnt_item[j : j + seq_size], axis=1).tolist())
        train_cnt_y.append(train_cnt_item[j + 1 : j + seq_size + 1])
        train_round_x.append(np.expand_dims(train_round_item[j : j + seq_size], axis=1).tolist())
        train_round_y.append(train_round_item[j + 1 : j + seq_size + 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(5):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_cnt_x, Y: train_cnt_y})
        prev_seq = train_cnt_x[-1]
        predict = []
        for j in range(30):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prediction_cnt[j][cursor] = predict[j]
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(5):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_round_x, Y: train_round_y})
        prev_seq = train_round_x[-1]
        predict = []
        for j in range(30):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prediction_round[j][cursor] = predict[j]
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
    print("Prediction", (cursor / 541555) * 100)
    if i % 10 == 0:
        print("Time:",time.time() - start_time)

print("Write result to output.csv")
csvfile = open('agent_rank_pagerank.csv', 'w', newline = '')
writer = csv.writer(csvfile)
writer.writerow(['day_id', 'sale_nbr', 'buy_nbr', 'cnt', 'round'])
data = []
for i in range(30):
    for j in range(541555):
        day = i + 1
        cursor = j + 1
        data.append(day, parent_sale[j], parent_buy[j], 
                    int(prediction_cnt[cursor]), int(prediction_round[cursor]))
writer.writerows(data)
csvfile.close()
print("Write success")
'''
print("Time:",time.time() - start_time)