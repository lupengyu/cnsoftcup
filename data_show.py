#coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_file = 'agent_rank_pagerank.csv'
full_data = pd.read_csv(data_file)
point_data = np.array(full_data['point'])
day_data = np.array(full_data['day'])

show_data = []
i = 0
while i >= 0:
    if(day_data[i] == 1):
        show_data.append(point_data[i])
        i += 1
    else:
        break
print(show_data)
plt.hist(show_data)
plt.xlabel("agent")
plt.ylabel("point")

#plt.plot(point_data)
plt.show()