#-*- coding: utf-8 -*-
import os
import string
import math
import operator
import datetime
import time
from time import mktime
from datetime import datetime, timedelta
import numpy as np
print (os.getcwd())
elevents= open(os.getcwd() + '/streaming_full.txt', 'r').readlines()
acts = open(os.getcwd() + '/activities.txt', 'r').readlines()
sensors = open(os.getcwd() + '/sensors.txt', 'r').readlines()
output = open(os.getcwd() + '/hmm_window_all.txt', 'w')


#========step1. build activities list=======
acts_list=[]
for link in acts:
    ss = link.strip().split('\t')
    act =ss[0]
    acts_list.append(act)
    #print(act)
acts_count=len(acts_list)
print('build activities list')


last_states = {}

def quan(i, min_value=0, max_value=40, level=80):
    return int((i-min_value) * (level/(max_value-min_value)))

#========step2. build sensors list=======
sensors_list=[]
for link in sensors:
    ss = link.strip().split('\t')
    sensor =ss[0]
    if sensor[0] != "T":
        sensors_list.append(sensor)
        last_states[sensor] = False #quan(23.0) if sensor[0] == 'T'

#print (sensors_list)
print('build sensors list')
sensors_count=len(sensors_list)


activity_start = 0
current_act = None
timestamp = None
past_timestamp = None
act_id = None

xs = []
ys = []
temp_x = []
temp_y = []

for link in elevents:
    ss = link.strip().split('\t')
    timestamp = datetime.strptime(ss[0]+" "+ss[1],'%Y-%m-%d %H:%M:%S.%f')
    name = ss[2]
    value = ss[3]
    act_id = acts_list.index(ss[4])

    if current_act is None:
        current_act = act_id
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)

    elif current_act != act_id:
        # if act_id == 0: continue

        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(current_act)
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)
        current_act = act_id

    if timestamp is not None and past_timestamp is not None and (timestamp.hour == 0 and past_timestamp.hour == 23):
        temp_x.append((past_timestamp.hour + past_timestamp.minute/60.0 + past_timestamp.second/3600.0))
        temp_y.append(current_act)
        xs.append(temp_x)
        ys.append(temp_y)
        temp_x = []
        temp_y = []
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)
        current_act = act_id
        break


    past_timestamp = timestamp


xs.append(temp_x)
ys.append(temp_y)


activity_start = 0
current_act = None
timestamp = None
past_timestamp = None
act_id = None
for link in elevents:
    ss = link.strip().split('\t')
    timestamp = datetime.strptime(ss[0]+" "+ss[1],'%Y-%m-%d %H:%M:%S.%f')
    name = ss[2]
    value = ss[3]
    act_id = acts_list.index(ss[4])

    if current_act is None:
        current_act = act_id
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)

    elif current_act != act_id:
        if act_id == 0: continue

        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(current_act)
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)
        current_act = act_id

    if timestamp is not None and past_timestamp is not None and (timestamp.hour == 0 and past_timestamp.hour == 23):
        temp_x.append((past_timestamp.hour + past_timestamp.minute/60.0 + past_timestamp.second/3600.0))
        temp_y.append(current_act)
        xs.append(temp_x)
        ys.append(temp_y)
        temp_x = []
        temp_y = []
        temp_x.append((timestamp.hour + timestamp.minute/60.0 + timestamp.second/3600.0))
        temp_y.append(act_id)
        current_act = act_id
        break


    past_timestamp = timestamp


xs.append(temp_x)
ys.append(temp_y)

# print ys

from matplotlib import pyplot as plt
import numpy as np


plt.style.use('fivethirtyeight')


fig, ax = plt.subplots()

count = 0
for x, y in zip(xs, ys):
    ax.plot(x, y)
    count += 1
    # if count == 1:
    #     break

ax.set_yticks(np.arange(0, 36, 5))
ax.set_yticks(np.arange(0, 36, 1), minor=True)
ax.set_xticks(np.arange(0, 25, 1))
ax.grid(which='both')

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.set_title("'HH104'")

plt.yticks(range(33), acts_list, size="small")
plt.show()