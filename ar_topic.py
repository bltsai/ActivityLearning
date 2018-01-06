#-*- coding: utf-8 -*-
import datetime, time
from time import mktime
from datetime import datetime, timedelta
import numpy as np
np.set_printoptions(threshold=np.inf)
from operator import add
import lda

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

DAYSTAMP_START = "daystart"
TIMESTAMP_START = "timestart"
PARENT_ACT = "parentact"
START_SENSOR_ID = "startsensorid"
SENSORS = "sensors"

TRAIN_ITERATION = 1500 # default 1500

import sys
RATIO = int(sys.argv[1])/100.0
# For HH104
# WHITELIST = [125, 95, 119, 121, 115, 120, 102, 111, 99, 87, 110, 123, 76, 63, 107, 84, 66, 85, 91, 101, 88, 103, 93, 79, 94, 75, 81, 90, 89, 106]

# For CAIRO
# WHITELIST = [23, 4, 20, 21, 11, 13, 10, 22, 30, 5, 1, 12, 31, 6, 8, 2, 29, 0, 7, 19, 27, 16, 9, 28, 15, 14, 24, 17, 18, 26]
SELECTED_NUMBER = int(sys.argv[2])
WHITELIST = None

def getListFromLines(fname):
    ret = []
    with open(fname, "r") as f:
        for l in f:
            l = l.strip().split()
            ret.append(l[0])
    return ret

def indexOfPrefix(l, s):
    max_candidate = ""
    max_candidate_i = -1
    for index, value in enumerate(l):
        if value in s and len(value) > len(max_candidate):
            max_candidate = value
            max_candidate_i = index

    return max_candidate_i

def processing(i_fname, act_t, sensor_t, o_info, mutual_info, mutual_count):
    with open(i_fname, "r") as ifd:
        OTHER_ID = act_t.index("Other")

        activity_begin = False
        other_begin = False
        handle_end = False
        now_act_id = OTHER_ID
        act_dict = {}

        i = 0
        for line in ifd:
            i+=1
            if i > 142322:
                break
            l = line.strip().split()
            timestamp = l[0] + " " + l[1]
            sensor_id = sensor_t.index(l[2])
            sensor_value = l[3]

            # Two cases for activity begin:
            # First is Other activity which is hidden between the end of a known activity and the begin of another known one
            if (len(l) > 4 and "begin" in l[4] and other_begin):
                other_begin = False
                handle_end = True

            # Second is the end of a known activity
            elif (len(l) > 4 and  "end" in l[4]):
                tmp_id = indexOfPrefix(act_t, l[4])
                # if now_act_id != tmp_id:
                #     print("Error! end check act_id unmatched %s (now) and %s (read)\n\t%s" %(act_t[now_act_id], act_t[tmp_id], line))
                #     print("end check change to ", act_t[tmp_id])
                now_act_id = tmp_id
                handle_end = True

                # streaming output
                if "streaming" in o_info:
                    ofd = o_info["streaming"]["ofd"]
                    ofd.write("\t".join(l[:4]+[act_t[now_act_id]]) + "\n")

            # output activity window, lda feature, mutual information if Other activiy or a known activity ends
            if handle_end:

                handle_end = False
                if now_act_id not in act_dict:
                    print("Error! end now_act_id %s not found\n\t%s" % (str(now_act_id), line))

                else:
                    act_info = act_dict[now_act_id]
                    sensor_dict = act_info[SENSORS]
                    sensor_dict[sensor_id] = sensor_dict.get(sensor_id, 0) + 1

                    involved_sensor_count = len(sensor_dict)
                    event_count = sum(sensor_dict.values())

                    starttime = datetime.strptime(act_info[TIMESTAMP_START],'%Y-%m-%d %H:%M:%S.%f')
                    endtime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')
                    weekday = starttime.weekday()#如果是星期一，返回0；如果是星期2，返回1
                    timespan = (time.mktime(endtime.timetuple()) - time.mktime(starttime.timetuple())) / (12 * 3600)
                    time_start = starttime.hour + starttime.minute / 60
                    time_end = endtime.hour + endtime.minute / 60

                    if "window" in o_info:
                        func = o_info["window"]["func"]
                        ofd = o_info["window"]["ofd"]
                        func(ofd, act_t, now_act_id, time_start, time_end, timespan, act_info, sensor_id, involved_sensor_count, event_count)

                    if "lda" in o_info:
                        func = o_info["lda"]["func"]
                        ofd = o_info["lda"]["ofd"]
                        func(ofd, act_t, sensor_t, now_act_id, starttime, endtime, act_info)

                    if "pcainput" in o_info:
                        func = o_info["pcainput"]["func"]
                        ofd = o_info["pcainput"]["ofd"]
                        func(ofd, now_act_id, sensor_t, act_dict[now_act_id][SENSORS])

                    if now_act_id != OTHER_ID and "mutual" in o_info:
                        func = o_info["mutual"]["func"]
                        func(mutual_info, mutual_count, act_info)

                    # print(now_act_id, act_t[now_act_id], act_dict)
                    # print()
                    sensor_dict.clear()
                    del act_dict[now_act_id]

                    # Store the ending current activity
                    tmp_id = now_act_id
                    # Change current activity to be the previous/parent activity
                    now_act_id = act_info[PARENT_ACT]

                    # if no known activity is running, change the activity flag
                    # if now_act_id (previou/parent activity) is Other or tmp_id (current activity) is not Other, replace the current activity to be the non-Other activity
                    if len(act_dict) == 0:
                        if now_act_id == OTHER_ID or tmp_id != OTHER_ID:
                            now_act_id = tmp_id
                        activity_begin = False

                    # if now_act_id (previou/parent activity) is not one of the running activity, find the latest running one
                    elif now_act_id not in act_dict:
                        # print("find for lastest id %s" % act_t[now_act_id])

                        max_time = datetime.strptime('1990-01-01 00:00:00.0000','%Y-%m-%d %H:%M:%S.%f')
                        for tmp_id, tmp_info in act_dict.items():
                            tmp_time = tmp_info[TIMESTAMP_START]
                            tmp_time = datetime.strptime(tmp_time,'%Y-%m-%d %H:%M:%S.%f')
                            if max_time < tmp_time:
                                max_time = tmp_time
                                now_act_id = tmp_id
                        # print("found it is id %s" % act_t[now_act_id])


            # if it is handling the end of a known activity, continue to get next line
            if (len(l) > 4 and  "end" in l[4]):
                continue

            # if it is the beginning of a known activity or hidden Other activity, change the flag and initialize the activity info
            if (len(l) > 4 and "begin" in l[4]) or \
                (len(l) == 4 and (not activity_begin) and (not other_begin)):
                act_info = {TIMESTAMP_START: timestamp, PARENT_ACT: now_act_id, START_SENSOR_ID: sensor_id, SENSORS: {} }

                if len(l) > 4:
                    now_act_id = indexOfPrefix(act_t, l[4])
                    activity_begin = True
                else:
                    now_act_id = OTHER_ID
                    other_begin = True

                act_dict[now_act_id] = act_info
                # print("begin", now_act_id, act_t[now_act_id], act_dict.keys())

            # if it is in the middle of a known activity or Other activity, count the sensor event and write to the streaming output
            if activity_begin or other_begin:
                if "streaming" in o_info:
                    ofd = o_info["streaming"]["ofd"]
                    ofd.write("\t".join(l[:4]+[act_t[now_act_id]]) + "\n")
                try:
                    sensor_dict = act_dict[now_act_id][SENSORS]
                    sensor_dict[sensor_id] = sensor_dict.get(sensor_id, 0) + 1
                except Exception as e:
                    print(l)
                    print(activity_begin, other_begin)
                    print(act_dict)
                    raise e



def func_window(ofd, act_t, now_act_id, time_start, time_end, timespan, act_info, sensor_id, involved_sensor_count, event_count):
    # now_act_id = act_t[now_act_id]
    output_line = [str(now_act_id)]
    output_line.append(str(round(time_start,3)))
    output_line.append(str(round(time_end,3)))
    output_line.append(str(round(timespan,3)))
    tmp_id = act_info[PARENT_ACT]
    # tmp_id = act_t[tmp_id]
    output_line.append(str(tmp_id))
    output_line.append(str(act_info[START_SENSOR_ID]))
    output_line.append(str(sensor_id))
    output_line.append(str(involved_sensor_count))
    output_line.append(str(event_count))
    output_line = "\t".join(output_line) + "\n"

    ofd.write(output_line)

def func_pca_input(ofd, now_act_id, sensor_t, sensors):
    window_sensor_array = []
    for i in range(0, len(sensor_t)):
        window_sensor_array.append(sensors.get(i, 0))

    window_sensor_array.append(now_act_id)
    ofd.write(str(window_sensor_array)[1:-1] + "\n")

def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)

def sensor_selection(ifd, sensor_number, sensor_t, target_number,):
    sensor_matrics = []
    activity_array = []
    mutual_info = []
    entropy_info = []
    target_sensor_index = []
    selected_sensors = []
    selected_sensor_ids = []
    for l in ifd:
        l = l.strip().split(',')
        sensor_array = np.asfarray(l[:sensor_number])
        activity_array.append(float(l[sensor_number]))
        sensor_matrics.append(sensor_array)
    # svd = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
    # svd.fit(sensor_matrics, activity_array)
    # print(svd.explained_variance_ratio_)
    # print(svd.singular_values_)

    for i in range(sensor_number):
        entropy_info.append((entropy(np.array(np.array(sensor_matrics)[:,i]).astype(int)), i))
        mutual_info.append((mutual_info_score(np.array(sensor_matrics)[:,i], activity_array), i))

    print(entropy_info)

    print("Original sensor list %s" % (sensor_t))
    # select target number of sensor that maximize the sum of mutual_info and
    # minimize the lose of sum of entropy_info from total sensor_number
    mutual_info.sort(key=lambda tup: tup[0], reverse=True)
    for m in mutual_info:
        selected_sensors.append(sensor_t[m[1]])
        selected_sensor_ids.append(m[1])

    print("Sorted sensor list %s by importance" % (selected_sensors))
    print("Selected Sensor Ids %s" % selected_sensor_ids[:target_number])
    return selected_sensor_ids[:target_number]




def func_ldafeature(ofd, act_t, sensor_t, now_act_id, starttime, endtime, act_info):
    sensor_dict = act_info[SENSORS]
    weekday = starttime.weekday()#如果是星期一，返回0；如果是星期2，返回1

    time_start = starttime.hour#+start.minute/60
    time_end = endtime.hour#+end.minute/60

    output_line = [str(now_act_id)]

    count = 1
    for i in range(7):
        feature = 0 if weekday != i else 1
        output_line.append(str(count) + ":" + str(feature))
        count += 1

    for j in range(24):
        feature = 1 if j >= time_start and j <= time_end else 0
        output_line.append(str(count) + ":" + str(feature))
        count += 1

    for j in range(len(act_t)):
        feature = 0 if j != now_act_id else 1
        output_line.append(str(count) + ":" + str(feature))
        count += 1

    for i, sensor in enumerate(sensor_t):
        if (not isWhiteList(i)):
            feature = 0
        else:
            feature = 0 if i not in sensor_dict else sensor_dict[i]
        output_line.append(str(count) + ":" + str(feature))
        count += 1

    output_line = "\t".join(output_line) + "\n"

    ofd.write(output_line)

def func_mutual(mutual_info, mutual_count, act_info):
    mutual_count[0] += 1
    sensor_dict = act_info[SENSORS]
    involved_sensor_list = sensor_dict.keys()

    for row in involved_sensor_list:
        if (isWhiteList(row)):
            for col in involved_sensor_list:
                if (isWhiteList(col)):
                    mutual_info[row][col] += 1

def mutual_output(mutual_info, mutual_count, ofd):
    count = mutual_count[0]
    output_line = []
    for i, row in enumerate(mutual_info):
        output_line=[str(count)]
        for column in row:
            output_line.append(str(column))
        ofd.write("\t".join(output_line) + "\n")
        if i % 1000 == 0:
            ofd.flush()


def getWindowSize(window_fname, act_t, sensor_t):

    window_size = 0
    window_num = 0
    with open(window_fname, "r") as ifd:
        window_dict = {}
        for l in ifd:
            l = l.strip().split()
            act = l[0]
            starttime = float(l[1])
            endtime = float(l[2])
            timespan = float(l[3])
            parentact = int(l[4])
            startsensorid = l[5]
            lastsensorid = l[6]
            involved_sensor_count = float(l[7])
            event_count = float(l[8])

            tmp_list = [starttime, endtime, timespan, involved_sensor_count, event_count, 1]
            if act not in window_dict:
                window_dict[act] = tmp_list
            else:
                window_dict[act] = list(map(add, window_dict[act], tmp_list))
            window_num += 1

        total_sensor_count_per_act_window = 0
        for act in window_dict:
            window_count_per_act = window_dict[act][5]
            total_sensor_count_per_act_window += window_dict[act][3] / window_count_per_act

        window_size = int(total_sensor_count_per_act_window / len(act_t)) + 1

    return window_size, window_num

def trainTopicModel(lda_fname, window_num, act_t, sensor_t):
    n_samples = window_num #1180
    acts_count = len(act_t)
    sensors_count = len(sensor_t)
    n_features = 7+24+acts_count+sensors_count
    n_topics = acts_count
    X_train, y_train = load_svmlight_file(lda_fname)
    X_train = X_train.astype(int)

    act_array=[0 for i in range(n_topics)]
    topic_array=[0 for i in range(n_topics)]

    #Y_train=transpose(y_train)

    model = lda.LDA(n_topics=n_topics, n_iter=TRAIN_ITERATION, random_state=1)
    X_train = X_train.toarray()
    # print (X_train)
    # print (X_train.shape)

    model.fit(X_train)
    topic_word = model.topic_word_
    # print("training lda   end")
    Y_train = y_train.astype(int)
    for j in Y_train:
        topic_array[j] += 1 ##original

    #doc_array=[0 for i in range(n_samples)]
    #topic_array=[0 for i in range(n_topics)]

    topic_act = np.zeros((n_topics,n_topics))
    doc_topic = model.doc_topic_
    for i, row in enumerate(doc_topic):
        max1 = doc_topic[i].argmax()
        #doc_array[i]=max1  #learned
        #topic_array[max1]=topic_array[max]+1#
        topic_act[max1][Y_train[i]] += 1

    map1 = [0 for i in range(n_topics)]

    for i, row in enumerate(topic_act):
        index1 = topic_act[i].argmax()#
        map1[i] = index1#map

    for i in range(n_topics):
        j = map1[i]
        act_array[i] = topic_array[j] / n_samples
    # print(map1)
    del doc_topic
    return topic_word, act_array


def getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array):
    mutual_info.astype(float)
    mutual_info = mutual_info / mutual_count[0]

    sliding_window = []
    OTHER_ID = act_t.index("Other")
    parent_act = OTHER_ID
    run = 0
    for l in ifd:
        l = l.strip().split()
        if len(l) != 5: continue
        timestamp = l[0] + " " + l[1]
        if timestamp.find('.') < 0:
            timestamp += '.0000'
        sensor_id = sensor_t.index(l[2])
        sensor_value = l[3]
        act_id = indexOfPrefix(act_t, l[4])
        item = [timestamp, sensor_id, act_id]

        if (not isWhiteList(sensor_id) or act_id == OTHER_ID):
            continue
        if len(sliding_window) < window_size:
            sliding_window.append(item)

        else:
            parent_act = sliding_window[window_size-1][2]
            sliding_window.pop(0)
            sliding_window.append(item)
            endtime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')
            sensor_last_id = sensor_id
            now_act_id = item[2]
            involved_sensor_list = []
            for index in range(window_size):
                item = sliding_window[index]
                involved_sensor_list.append(item[1])
                if index == 0:
                    starttime = datetime.strptime(item[0],'%Y-%m-%d %H:%M:%S.%f')
                    sensor_start_id = item[1]
            weekday = starttime.weekday()
            timespan = (time.mktime(endtime.timetuple())-time.mktime(starttime.timetuple()))/(12*3600)

            time_start = starttime.hour + starttime.minute / 60
            time_end = endtime.hour + endtime.minute / 60

            lda_feature = []
            for i in range(7):
                feature = 0 if weekday != i else 1
                lda_feature.append(feature)

            for j in range(24):
                feature = 1 if j >= time_start and j <= time_end else 0
                lda_feature.append(feature)

            for j in range(len(act_t)):
                feature = 0 if j != now_act_id else j
                lda_feature.append(feature)

            for i, sensor in enumerate(sensor_t):
                feature = 0 if i not in involved_sensor_list else 1
                lda_feature.append(feature)

            tmp_list = []
            tmp_list.append(str(round((weekday+1)/7,3)))
            tmp_list.append(str(round(time_start/24,3)))
            tmp_list.append(str(round(time_end/24,3)))
            tmp_list.append(str(round(timespan,3)))
            tmp_list.append(str(round(parent_act/len(act_t),3)))
            tmp_list.append(str(round(sensor_start_id/len(sensor_t),3)))
            tmp_list.append(str(round(sensor_last_id/len(sensor_t),3)))
            tmp_list.append(str(round(len(involved_sensor_list)/len(sensor_t),3)))

            row = sensor_last_id
            for col, sensor in enumerate(sensor_t):
                value = 0
                if col in involved_sensor_list and isWhiteList(col):
                    value = mutual_info[row][col]
                tmp_list.append(str(round(value,3)))

            for j, row in enumerate(topic_word):
                temp = act_array[j] * np.dot(lda_feature, np.log1p(row))
                tmp_list.append(str(round(temp, 5)))

            del lda_feature

            output_line = [str(now_act_id)]
            for i, feature in enumerate(tmp_list, start = 1):
                output_line.append("%d:%s"%(i,feature))
            output_line = "\t".join(output_line) + "\n"
            ofd.write(output_line)

            if now_act_id != OTHER_ID:
                parent_act = now_act_id

            run += 1
            if run % 1000 == 0:
                ofd.flush()


def classify(n_classes=None):
    X_train, y_train = load_svmlight_file("trainstream.txt")
    X_test, y_test = load_svmlight_file("teststream.txt", n_features=X_train.shape[1])
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=10,max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(X_train, y_train)
    t1 = time.time()
    print ("training time running : %s seconds" % str(t1-t0))
    t0 = time.time()
    y_pred= clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    t1 = time.time()
    print ("testing time running : %s seconds" % str(t1-t0))
    print("\naccuracy: %.4f %%\n"% accuracy)


    if n_classes is not None:
        conmatrix_all=confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
        tsp_conmatrix=np.transpose(conmatrix_all)
        p=[]
        for i,row in enumerate(tsp_conmatrix):

            if sum(row)>0:
                x=row[i]/sum(row)
            else:
                x=-1
            p.append(x)
        print("Precision for all classes")
        print(p)

        r=[]
        for i,row in enumerate(conmatrix_all):

            if sum(row)>0:
                x=row[i]/sum(row)
            else:
                x=-1
            r.append(x)

        print("Recall for all classes")
        print(r)


    conmatrix=confusion_matrix(y_test, y_pred)
    tsp_conmatrix=np.transpose(conmatrix)
    print("Confusion Matrix:")
    print(conmatrix)
    p=[]
    for i,row in enumerate(tsp_conmatrix):

        if sum(row)>0:
            x=row[i]/sum(row)
        else:
            x=1
        p.append(x)

    print("Precision for appeared classes")
    print(p)

    r=[]
    for i,row in enumerate(conmatrix):

        if sum(row)>0:
            x=row[i]/sum(row)
        else:
            x=1
        r.append(x)

    print("Recall for appeared classes")
    print(r)

    f=[]
    for i in range(len(p)):
        if p[i]*r[i]>0:
            y=2*p[i]*r[i]/(p[i]+r[i])
        else:
            y=0
        f.append(y)
    print("f1_score array")
    print(f)
    print("\nf1_score %f\n" % (sum(f)/len(f)))

    importances = clf.feature_importances_

    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    print("std:")
    print(std)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(X_train.shape[1]), importances[indices],
           #color="r", yerr=std[indices], align="center")
    #plt.xticks(range(X_train.shape[1]), indices)
    #plt.xlim([-1, X_train.shape[1]])
    #plt.show()


# Prof. Yan's original processing task
def task_original():
    act_t_fname = "activities.txt"
    sensor_t_fname = "sensors.txt"
    data_fname = "data.txt"
    streaming_fname = "streaming.txt"
    window_fname = "window.txt"
    lda_fname = "lda.txt"
    mutual_fname = "mumtrix.txt"
    featurestream_fname = "featurestream.txt"
    pca_input_fname = "pcainput.txt"

    print("Read activity types...", end="", flush=True)
    act_t = getListFromLines(act_t_fname)
    print("Done")

    print("Read sensor types...", end="", flush=True)
    sensor_t = getListFromLines(sensor_t_fname)
    print("Done")

    selected_sensors = []
    print("Sensor Selection on the pcainput...", end="", flush=True)
    with open(pca_input_fname, "w") as ofd5:
        o_info = {}
        o_info["pcainput"] = {"ofd":ofd5, "func":func_pca_input}
        processing(data_fname, act_t, sensor_t, o_info, None, None)

    with open(pca_input_fname, "r") as ifd:
        selected_sensors = sensor_selection(ifd, len(sensor_t), sensor_t, 30)

    global WHITELIST
    WHITELIST = set(selected_sensors[:SELECTED_NUMBER])
    print("Done")
    print("Selected:")
    print(selected_sensors)
    print("Whitelist:")
    print(WHITELIST)

    print("Process streaming label, activity window, LDA feature, and mutual info...", end="", flush=True)
    with open(streaming_fname, "w") as ofd1, open(window_fname, "w") as ofd2, open(lda_fname, "w") as ofd3, open(mutual_fname, "w") as ofd4:
        o_info = {}
        o_info["streaming"] = {"ofd":ofd1}
        o_info["window"] = {"ofd":ofd2, "func":func_window}
        o_info["lda"] = {"ofd":ofd3, "func":func_ldafeature}
        o_info["mutual"] = {"func": func_mutual}
        mutual_info = np.zeros((len(sensor_t), len(sensor_t)))
        mutual_count = [0]
        processing(data_fname, act_t, sensor_t, o_info, mutual_info, mutual_count)
        mutual_output(mutual_info, mutual_count, ofd4)
    print("Done")

    print("Calculate window size...", end="", flush=True)
    window_size, window_num = getWindowSize(window_fname, act_t, sensor_t)
    print("Done")
    print("window size is %d" % window_size)

    print("Train Topic Model...", flush=True)
    topic_word, act_array = trainTopicModel(lda_fname, window_num, act_t, sensor_t)
    print("Train Topic Model...Done", flush=True)

    print("Process sliding window...", end="", flush=True)
    with open(featurestream_fname, "w") as ofd, open(streaming_fname, "r") as ifd:
        getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    print("Done")

    # Get total line count
    with open(featurestream_fname, "r") as ifd:
        for i, _ in enumerate(ifd): pass
        line_count = i + 1

    # Splite feature stream to 75 training and 25 testing
    line_count = int(round(line_count * RATIO))
    with open(featurestream_fname, "r") as ifd, open("trainstream.txt", "w") as ofd, open("teststream.txt", "w") as ofd2:
        for i, l in enumerate(ifd):
            if i < line_count:
                ofd.write(l)
            else:
                ofd2.write(l)

    classify(len(act_t))

def task_analysis():
    act_t_fname = "activities.txt"
    sensor_t_fname = "sensors.txt"
    data_fname = "data.txt"
    streaming_fname = "streaming.txt"
    window_fname = "window.txt"
    lda_fname = "lda.txt"
    mutual_fname = "mumtrix.txt"
    featurestream_fname = "featurestream.txt"
    pca_input_fname = "pcainput.txt"

    print("Read activity types...", end="", flush=True)
    act_t = getListFromLines(act_t_fname)
    print("Done")

    print("Read sensor types...", end="", flush=True)
    sensor_t = getListFromLines(sensor_t_fname)
    print("Done")

    # selected_sensors = []
    # print("Sensor Selection on the pcainput...", end="", flush=True)
    # with open(pca_input_fname, "w") as ofd5:
    #     o_info = {}
    #     o_info["pcainput"] = {"ofd":ofd5, "func":func_pca_input}
    #     processing(data_fname, act_t, sensor_t, o_info, None, None)

    # with open(pca_input_fname, "r") as ifd:
    #     selected_sensors = sensor_selection(ifd, len(sensor_t), sensor_t, 30)

    # # selected_sensors = range(len(sensor_t))

    # global WHITELIST
    # WHITELIST = set(selected_sensors)
    # print("Done")
    # print("Selected:")
    # print(selected_sensors)
    # print("Whitelist:")
    # print(WHITELIST)

    # print("Process streaming label, activity window, LDA feature, and mutual info...", end="", flush=True)
    # with open(streaming_fname, "w") as ofd1, open(window_fname, "w") as ofd2, open(lda_fname, "w") as ofd3, open(mutual_fname, "w") as ofd4:
    #     o_info = {}
    #     o_info["streaming"] = {"ofd":ofd1}
    #     o_info["window"] = {"ofd":ofd2, "func":func_window}
    #     o_info["lda"] = {"ofd":ofd3, "func":func_ldafeature}
    #     o_info["mutual"] = {"func": func_mutual}
    #     mutual_info = np.zeros((len(sensor_t), len(sensor_t)))
    #     mutual_count = [0]
    #     processing(data_fname, act_t, sensor_t, o_info, mutual_info, mutual_count)
    #     mutual_output(mutual_info, mutual_count, ofd4)
    # print("Done")

    # print("Calculate window size...", end="", flush=True)
    # window_size, window_num = getWindowSize(window_fname, act_t, sensor_t)
    # print("Done")
    # print("window size is %d" % window_size)

    # print("Train Topic Model...", flush=True)
    # topic_word, act_array = trainTopicModel(lda_fname, window_num, act_t, sensor_t)
    # print("Train Topic Model...Done", flush=True)

    # print("Process sliding window...", end="", flush=True)
    # with open(featurestream_fname, "w") as ofd, open(streaming_fname, "r") as ifd:
    #     getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    # print("Done")

    # Get total line count
    with open(featurestream_fname, "r") as ifd:
        for i, _ in enumerate(ifd): pass
        line_count = i + 1

    # Splite feature stream to 75 training and 25 testing
    line_count = int(round(line_count * RATIO))
    train_act_dist = {}
    all_act_dist = {}
    with open(featurestream_fname, "r") as ifd:
        for i, l in enumerate(ifd):
            l = l.strip().split()
            act_id = int(l[0])
            if i < line_count:
                train_act_dist[act_id] = train_act_dist.get(act_id, 0) + 1
            all_act_dist[act_id] = all_act_dist.get(act_id, 0) + 1

    for i in range(len(act_t)):
        count = train_act_dist.get(i, 0)
        total = all_act_dist.get(i, 0)
        print(i, act_t[i], count, total)


def task_baseline():
    act_t_fname = "activities.txt"
    sensor_t_fname = "sensors.txt"
    data_fname = "data.txt"
    streaming_fname = "streaming.txt"
    window_fname = "window.txt"
    lda_fname = "lda.txt"
    mutual_fname = "mumtrix.txt"
    featurestream_fname = "featurestream.txt"

    print("Read activity types...", end="", flush=True)
    act_t = getListFromLines(act_t_fname)
    print("Done")

    print("Read sensor types...", end="", flush=True)
    sensor_t = getListFromLines(sensor_t_fname)
    print("Done")

    print("Prepare training data...", end="", flush=True)
    with open(streaming_fname, "w") as ofd1:
        o_info = {}
        o_info["streaming"] = {"ofd":ofd1}
        mutual_info = None
        mutual_count = None
        processing(data_fname, act_t, sensor_t, o_info, mutual_info, mutual_count)

    # Get the timestamp of the last second of the date when is 75% of overall timespan among the data
    with open(data_fname, "r") as ifd:
        mintime = None
        maxtime = None
        for l in ifd:
            l = l.strip().split()
            timestamp = l[0] + " " + l[1]
            if mintime is None:
                mintime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')
        maxtime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')

        datediff = int(round((maxtime.date() - mintime.date()).days * 0.75))-1
        endtime = datetime.combine(mintime.date() + timedelta(days=datediff), datetime.max.time())

    # Split the streaming based on the timestamp above
    with open(streaming_fname, "r") as ifd, open("stream_training.txt", "w") as ofd, open("stream_testing.txt", "w") as ofd2:
        i = 0
        for line in ifd:
            i+=1
            l = line.strip().split()
            timestamp = l[0] + " " + l[1]
            if timestamp.find('.') < 0:
                timestamp += '.0000'

            print(timestamp)
            if datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f') <= endtime:
                ofd.write(line)
            else:
                ofd2.write(line)

    # Split out the training part of the data within the 75% timespan
    with open(data_fname, "r") as ifd, open("data_training.txt", "w") as ofd:
        i = 0
        for line in ifd:
            i+=1
            l = line.strip().split()
            timestamp = l[0] + " " + l[1]
            if timestamp.find('.') < 0:
                timestamp += '.0000'
            print("%d %s" % (i, timestamp))
            if datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f') <= endtime:
                ofd.write(line)
    print("Done")

    print("Process activity window, LDA feature, and mutual info...", end="", flush=True)
    with open(window_fname, "w") as ofd2, open(lda_fname, "w") as ofd3, open(mutual_fname, "w") as ofd4:
        o_info = {}
        o_info["window"] = {"ofd":ofd2, "func":func_window}
        o_info["lda"] = {"ofd":ofd3, "func":func_ldafeature}
        o_info["mutual"] = {"func": func_mutual}
        mutual_info = np.zeros((len(sensor_t), len(sensor_t)))
        mutual_count = [0]
        processing("data_training.txt", act_t, sensor_t, o_info, mutual_info, mutual_count)
        mutual_output(mutual_info, mutual_count, ofd4)
    print("Done")

    print("Calculate window size...", end="", flush=True)
    window_size, window_num = getWindowSize(window_fname, act_t, sensor_t)
    print("Done")
    print("window size is %d" % window_size)

    print("Train Topic Model...", flush=True)
    topic_word, act_array = trainTopicModel(lda_fname, window_num, act_t, sensor_t)
    print("Train Topic Model...Done", flush=True)


    print("Process training sliding window...", end="", flush=True)
    with open("trainstream.txt", "w") as ofd, open("stream_training.txt", "r") as ifd:
        getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    print("Done")

    print("Process testing sliding window...", end="", flush=True)
    with open("teststream.txt", "w") as ofd, open("stream_testing.txt", "r") as ifd:
        getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    print("Done")

    classify()

# whitelist_30 = {17, 26, 22, 28, 27, 21, 15, 23, 29, 3, 38, 25, 11, 6, 13, 8, 5, 30, 34, 16, 12, 7, 4, 14, 24, 35, 36, 10, 18, 20}
# whitelist_20 = {17, 26, 22, 28, 27, 21, 15, 23, 29, 3, 38, 25, 11, 6, 13, 8, 5, 30, 34, 16}
# whitelist_10 = {17, 26, 22, 28, 27, 21, 15, 23, 29, 3}
# whitelist_5 = {17, 26, 22, 28, 27}


def isWhiteList(sensor_id):
    return sensor_id in WHITELIST

def isBlacklist(act_t, sensor_t, l):


    timestamp = l[0] + " " + l[1]
    if timestamp.find('.') < 0:
        timestamp += '.0000'
    sensor_id = sensor_t.index(l[2])
    sensor_value = l[3]
    act_id = indexOfPrefix(act_t, l[4])

    ### TODO: filter the sensor event ###

    return False



def task_testing_blacklist():
    act_t_fname = "activities.txt"
    sensor_t_fname = "sensors.txt"
    data_fname = "data.txt"
    streaming_fname = "streaming.txt"
    window_fname = "window.txt"
    lda_fname = "lda.txt"
    mutual_fname = "mumtrix.txt"
    featurestream_fname = "featurestream.txt"

    print("Read activity types...", end="", flush=True)
    act_t = getListFromLines(act_t_fname)
    print("Done")

    print("Read sensor types...", end="", flush=True)
    sensor_t = getListFromLines(sensor_t_fname)
    print("Done")

    print("Prepare training data...", end="", flush=True)
    with open(streaming_fname, "w") as ofd1:
        o_info = {}
        o_info["streaming"] = {"ofd":ofd1}
        mutual_info = None
        mutual_count = None
        processing(data_fname, act_t, sensor_t, o_info, mutual_info, mutual_count)

    # Get the timestamp of the last second of the date when is 75% of overall timespan among the data
    with open(data_fname, "r") as ifd:
        mintime = None
        maxtime = None
        for l in ifd:
            l = l.strip().split()
            timestamp = l[0] + " " + l[1]
            if mintime is None:
                mintime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')
        maxtime = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f')

        datediff = int(round((maxtime.date() - mintime.date()).days * 0.75))-1
        endtime = datetime.combine(mintime.date() + timedelta(days=datediff), datetime.max.time())

    # Split the streaming based on the timestamp above
    # For testing data, block the sensor event based on blacklist
    with open(streaming_fname, "r") as ifd, open("stream_training.txt", "w") as ofd, open("stream_testing.txt", "w") as ofd2:
        for line in ifd:
            l = line.strip().split()
            timestamp = l[0] + " " + l[1]
            if timestamp.find('.') < 0:
                timestamp += '.0000'
            if datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f') <= endtime:
                ofd.write(line)
            elif not isBlacklist(act_t, sensor_t, l):
                ofd2.write(line)

    # Split out the training part of the data within the 75% timespan
    with open(data_fname, "r") as ifd, open("data_training.txt", "w") as ofd:
        for line in ifd:
            l = line.strip().split()
            timestamp = l[0] + " " + l[1]
            if timestamp.find('.') < 0:
                timestamp += '.0000'
            if datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f') <= endtime:
                ofd.write(line)
    print("Done")

    print("Process activity window, LDA feature, and mutual info...", end="", flush=True)
    with open(window_fname, "w") as ofd2, open(lda_fname, "w") as ofd3, open(mutual_fname, "w") as ofd4:
        o_info = {}
        o_info["window"] = {"ofd":ofd2, "func":func_window}
        o_info["lda"] = {"ofd":ofd3, "func":func_ldafeature}
        o_info["mutual"] = {"func": func_mutual}
        mutual_info = np.zeros((len(sensor_t), len(sensor_t)))
        mutual_count = [0]
        processing("data_training.txt", act_t, sensor_t, o_info, mutual_info, mutual_count)
        mutual_output(mutual_info, mutual_count, ofd4)
    print("Done")

    print("Calculate window size...", end="", flush=True)
    window_size, window_num = getWindowSize(window_fname, act_t, sensor_t)
    print("Done")
    print("window size is %d" % window_size)

    print("Train Topic Model...", flush=True)
    topic_word, act_array = trainTopicModel(lda_fname, window_num, act_t, sensor_t)
    print("Train Topic Model...Done", flush=True)


    print("Process training sliding window...", end="", flush=True)
    with open("trainstream.txt", "w") as ofd, open("stream_training.txt", "r") as ifd:
        getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    print("Done")

    print("Process testing sliding window...", end="", flush=True)
    with open("teststream.txt", "w") as ofd, open("stream_testing.txt", "r") as ifd:
        getFeatureStream(ifd, ofd, window_size, act_t, sensor_t, mutual_info, mutual_count, topic_word, act_array)
    print("Done")

    classify()

if __name__ == "__main__":
    # task_original()
    task_analysis()
    # task_baseline()
    # task_testing_blacklist()
