#coding=utf-8
import pandas as pd
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import time
import warnings
import csv
warnings.filterwarnings("ignore")

#读取数据并按照类别分类
#5831365 rows*5 columns
in_file = 'sales_sample_20170310.csv'
print("Reading the data from", in_file)
full_data = pd.read_csv(in_file)
day_id_data = np.array(full_data['day_id'])
sale_nbr_data = np.array(full_data['sale_nbr'])
buy_nbr_data = np.array(full_data['buy_nbr'])
cnt_data = np.array(full_data['cnt'])
round_data = np.array(full_data['round'])

def Graph_build():
    cursor = 0
    days = []
    gents = []
    points = []
    ranks = []
    for i in range(91):#91
        day = i + 1
        print("Calculating day", day)
        G = nx.DiGraph()
        judge = True
        while judge:
            if cursor == 5831365:
                judge = False
                break
            if day_id_data[cursor] != day:
                judge = False
            else:
                G.add_weighted_edges_from([(sale_nbr_data[cursor],
                                            buy_nbr_data[cursor],
                                            round_data[cursor])])
                cursor += 1
        #do some things
        weights = nx.pagerank(G, alpha=0.85)
        rank = sorted(weights.items(), key=lambda e:e[1], reverse=True)
        #print(rank.__class__)
        biggest = rank[1][1];
        paiming = 1
        judge = False
        for i in rank:
            if judge == True:
                days.append(day)
                gents.append(i[0])
                points.append(100 * i[1] / biggest)
                ranks.append(paiming)
                paiming += 1
            judge = True
        G.clear()
    csvfile = open('agent_rank_pagerank.csv', 'w', newline = '')
    writer = csv.writer(csvfile)
    writer.writerow(['day', 'gent', 'point', 'rank'])
    data = []
    i = len(days)
    for j in range(i):
        data.append([days[j], gents[j], points[j], ranks[j]])
    writer.writerows(data)
    csvfile.close()
    
def situation():
    print("Calculating the situation of every agents")
    start_time = time.time()
    Graph_build()
    print("Time spend:",time.time() - start_time)
    
situation()