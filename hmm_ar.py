from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import time, os
import matplotlib.pyplot as plt
import numpy as np
import seqlearn
import seqlearn.hmm
import seqlearn.evaluation
from seqlearn.evaluation import whole_sequence_accuracy, SequenceKFold

def round(X_train, y_train, train_lengths, X_test, y_test, test_lengths):

    t0 = time.time()
    clf = seqlearn.hmm.MultinomialHMM()
    clf.fit(X_train, y_train, train_lengths)
    t1 = time.time()
    print ("training: %s seconds" % str(t1-t0))
    # print(clf.classes_)

    t0 = time.time()
    y_pred= clf.predict(X_test)
    # print(y_pred)
    # print(y_test)
    count = 0
    for i in xrange(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1

    # accuracy = whole_sequence_accuracy(y_test, y_pred, test_lengths)
    accuracy = count / (float)(len(y_test))
    t1 = time.time()
    print ("testing: %s seconds" % str(t1-t0))
    print("accuracy: %0.4f%%"% (accuracy*100))

    # exit()
    # print (clf.intercept_trans_)

    print " ".join(["Class", "Truth", "Positive", "True_Positive", "False_Positivie", "Precision", "Recall"])

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    for class_id in xrange(33):
        idx = (y_test == class_id)
        class_y_pred = y_pred[idx]
        class_y_test = y_test[idx]

        accuracy = 0
        count = 0
        if class_y_test.shape[0] != 0:
            for i in xrange(len(class_y_test)):
                if class_y_test[i] == class_y_pred[i]:
                    count += 1
            accuracy = count / (float)(class_y_test.shape[0])

        temp = y_pred[(y_pred == class_id)]

        # print "Class " + str(class_id) + " True " + str(class_y_test.shape[0]) + " P " + str(temp.shape[0]) + " TP " + str(count) + " FP " + str(temp.shape[0]-count) + " precision {:.4f}".format(accuracy) + " recall {:.4f}".format(count/float(class_y_test.shape[0]) if class_y_test.shape[0] else 0)
        t = str(class_y_test.shape[0])
        p = str(temp.shape[0])
        tp = str(count)
        fp = str(temp.shape[0]-count)
        precision = "{:.4f}".format(count/(float)(temp.shape[0]) if temp.shape[0] else 0)
        recall = "{:.4f}".format(accuracy)
        print " ".join([str(class_id), t, p, tp, fp, precision, recall])

    # print("confusion_matrix:")

    # print(confusion_matrix(y_test, y_pred))

    # conmatrix=confusion_matrix(y_test, y_pred)


# X, y = load_svmlight_file(os.getcwd() + "/hmm_window_all.txt")

# count = 0
# n_samples = X.shape[0]
# sequence_lenght = 15
# lengths = [sequence_lenght] * int(n_samples/sequence_lenght)
# if n_samples % sequence_lenght != 0:
#     lengths += [n_samples % sequence_lenght]
# fold = SequenceKFold(lengths, n_folds=4)
# for train_indices, train_lengths, test_indices, test_lengths in fold:
#     print "Round " + str(count)
#     count += 1
#     round(X[train_indices], y[train_indices], train_lengths, X[test_indices], y[test_lengths], test_lengths)


# X_train, y_train = load_svmlight_file(os.getcwd() + "/hmm_window.txt")
# X_test, y_test = load_svmlight_file( os.getcwd() + "/hmm_window_test.txt", n_features=X_train.shape[1])

def readfile(filename):
    words = []
    X = []
    y = []
    with open(filename) as f:
        lines = f.readlines()
        # for line in lines:
        #     tokens = line.split()
        #     words += tokens[1:]
        #     y.append(tokens[0])

        # vocab, identities = np.unique(words, return_inverse = True)
        # X_full = (identities.reshape(-1, 1) == np.arange(len(vocab)))

        # X = []
        # for line in lines:
        #     end_index = len(line.split()[1:])
        #     # print(end_index)
        #     # print(X_full[:end_index])
        #     X_part = np.logical_or.reduce(X_full[:end_index])
        #     # print(X_part)
        #     X.append(X_part)
        #     X_full = X_full[end_index:]
        for line in lines:
            l = eval(line)
            X.append(l[1:])
            y.append(l[0])

    X = np.array(X).astype(int)
    return X, y
X, y = readfile(os.getcwd() + "/hmm_window_all.txt")
# print X_train, y_train, length_train, X_test, y_test, length_test
# X_test, y_test, length_test = readfile(os.getcwd() + "/hmm_window_test.txt")

line = 6640
train_index = int(6640*.75)
test_num = line - train_index
round(X[:train_index], y[:train_index], [train_index], X[train_index:], y[train_index:], [test_num])
# round(X[:912], y[:912], [912], X[912:], y[912:], [303])