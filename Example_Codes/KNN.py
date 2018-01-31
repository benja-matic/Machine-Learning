import numpy as np
import matplotlib.pyplot as plt
import time as TIME
import scipy.spatial as ss
from sklearn.datasets import load_iris
#assume our vectors are numpy arrays
def euclidean_distance(v1, v2):
    distance = v1 - v2
    ip = np.inner(distance, distance)
    return np.sqrt(ip)


#takes k, number of classes, labeled data, unlabeled data, function handle for distance
#data should be nXp, where n is number of data points, p is number of features
#all p features are assumed to be continuous variables so we can think of each row as a p-dimensional vector
#labels should be 1D array of length p, assumed to take on dummy values of 0:c

def knn_vdist(k, c, data, labels, new_data, distance):
    N1 = np.shape(data)[0]
    N2 = np.shape(new_data)[0]
    new_labels = np.zeros(N2)
    #brute force for now
    for i in range(N2):
        d = np.zeros(N1)
        for j in range(N1):
            d[j] = distance(data[j,:], new_data[i,:])
        x = np.argsort(d)[:k] #indices in d, data rows, or labels, for the k nearest neighbors
        classes = np.zeros(c) #the value in labels points back to the index in classes
        for neighbor in x:
            classes[labels[neighbor]] += 1
        new_labels[i] = np.argmax(classes)
    return new_labels

x = range(len(data['data']))
np.random.shuffle(x)
c = len(set(data['target']))
train_data = data['data'][x[:75]]
test_data = data['data'][x[75:]]
train_labels = data['target'][x[:75]]
new_labels = knn_vdist(5, 3, train_data, train_labels, test_data, euclidean_distance)

correct = 0.
for i in range(len(new_labels)):
    check  = new_labels[i] - data['target'][x[75:]][i]
    if check == 0:
        correct +=1
print "got ", 100.*correct/75., "is the percent accuracy"
