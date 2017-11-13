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
# print (os.getcwd())
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
# print('build activities list')


last_states = {}

def quan(i, min_value=0, max_value=40, level=80):
    return int((i-min_value) * (level/(max_value-min_value)))

#========step2. build sensors list=======
sensors_list=[]
for link in sensors:
    ss = link.strip().split('\t')
    sensor =ss[0]
    sensors_list.append(sensor)
    last_states[sensor] = False

#print (sensors_list)
# print('build sensors list')
sensors_count=len(sensors_list)

# ######### Time window Method
# start_time = None
# time_delta = timedelta(seconds=4)

# for link in elevents:
#     ss = link.strip().split('\t')
#     timestamp = datetime.strptime(ss[0]+" "+ss[1],'%Y-%m-%d %H:%M:%S.%f')
#     if start_time is None: start_time = timestamp
#     name = ss[2]
#     value = ss[3]
#     act_id = acts_list.index(ss[4])
#     last_states[name] = quan(float(value)) if name[0] == "T" else 0 if value == "OFF" else 1
#     while timestamp - start_time > time_delta:
#         output_line = str(act_id)
#         for i in xrange(sensors_count):
#             state = last_states[sensors_list[i]]
#             output_line += "\t" + str(i) + ":" + str(state)

#         output.writelines(output_line+ '\n')
#         start_time += time_delta
#         if start_time >= timestamp or timestamp - start_time < time_delta:
#             start_time = timestamp
#             break

activity_start = 0
current_act = None
previos_act = None
timestamp = None
act_id = None
output_lines = []

RECOGNITION = False
EXCLUDE_OTHER = False

import sys
feature_selection = int(sys.argv[1])
print("FeatureSelection: {0:06b} = {1:02d}".format(feature_selection, feature_selection))

for link in elevents:
    ss = link.strip().split('\t')
    timestamp = datetime.strptime(ss[0]+" "+ss[1],'%Y-%m-%d %H:%M:%S.%f')
    name = ss[2]
    value = ss[3]
    act_id = acts_list.index(ss[4])

    if current_act is None:
        current_act = act_id
        previos_act = act_id
        output_line = []

    elif current_act == act_id:
        if value == "OFF" or value == "CLOSE":
            pass
        elif value[0] == "O":
            last_states[name] = True
        else:
            last_states[name] = True #eval(value)

    else: # !=
        if RECOGNITION:
            output_line.append(current_act) ## Activity Recognition
        else:
            output_line.append(act_id) ## Activity Prediction

        if (feature_selection >> 4) & 1 == 1:
            for i in xrange(sensors_count):
                output_line.append(last_states[sensors_list[i]])

        if (feature_selection >> 3) & 1 == 1:

            for i in xrange(12):
                output_line.append(timestamp.month == i)

        if (feature_selection >> 2) & 1 == 1:

            for i in xrange(31):
                output_line.append(timestamp.day == i)

        if (feature_selection >> 1) & 1 == 1:

            for i in xrange(24):
                output_line.append(timestamp.hour == i)

        if (feature_selection & 1) == 1:

            for i in xrange(acts_count):
                if RECOGNITION:
                    output_line.append(previos_act == i)
                else:
                    if current_act == 0:
                        output_line.append(previos_act == i)
                    else:
                        output_line.append(current_act == i)
                    # output_line.append(current_act == i)

        output_lines.append(output_line)
        previos_act = current_act
        current_act = act_id # if act_id != 0 else current_act
        output_line = []
        last_states = last_states.fromkeys(last_states, False)

## Excluding Other Activity Class by finding the next non-Other Class
if EXCLUDE_OTHER:
    for i, output_line in enumerate(output_lines):
        new_act_id = None
        if output_line[0] == 0:
            # for j in xrange(i-1, -1, -1):
            for j in xrange(i+1, len(output_lines)):
                if output_lines[j][0] != 0:
                    new_act_id = output_lines[j][0]
                    break
            output_lines[i][0] = new_act_id



for output_line in output_lines:
    output.writelines(str(output_line) + '\n')