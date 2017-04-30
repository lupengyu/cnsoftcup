#coding=utf-8
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
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
for i in range(91):
    for j in range(1000):
        cursor = j + 1
        train_cnt[i][j] = cnt_matrix[i][cursor]
        train_round[i][j] = round_matrix[i][cursor]
train_cnt_x = train_cnt[:90]
train_cnt_y = train_cnt[1:]
train_round_x = train_round[:90]
train_round_y = train_round[1:]
import tensorflow as tf
n_input_layer = 1000  # 输入层
n_layer_1 = 1000*3     # hide layer
n_layer_2 = 1000*4    # hide layer
n_layer_3 = 1000*2     # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层
n_output_layer = 1000   # 输出层

X = tf.placeholder('float', [None, 1000])
Y = tf.placeholder('float', [None, 1000])

layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
layer_3_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_layer_3])), 'b_':tf.Variable(tf.random_normal([n_layer_3]))}
layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    
layer_1 = tf.add(tf.matmul(X, layer_1_w_b['w_']), layer_1_w_b['b_'])
layer_1 = tf.nn.relu(layer_1)
'''
layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
layer_2 = tf.nn.relu(layer_2 )
layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
layer_3 = tf.nn.relu(layer_3 )
'''
layer_output = tf.add(tf.matmul(layer_1, layer_output_w_b['w_']), layer_output_w_b['b_'])

cost_func = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(layer_output,1e-10)))
#cost_func = tf.reduce_mean(tf.square(predict - Y))
optimizer = tf.train.AdamOptimizer(1).minimize(cost_func)
#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost_func)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1000):
        _, c = session.run([optimizer, cost_func], feed_dict={X:train_round_x,Y:train_round_y})
        if i%10 == 0: 
            print(i, ' : ', c)
    prev_seq = train_round[-1].reshape((1,1000))
    predict_round = []
    for j in range(30):
        pre_round = session.run(layer_output, feed_dict={X:prev_seq})
        predict_round.append(pre_round)
        prev_seq = pre_round
normalized_data = []
for i in range(91):
    normalized_data.append(round_matrix[i][50])
predict = []
for i in range(30):
    predict.append(predict_round[0][0][49])
plt.figure()
plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
plt.show()