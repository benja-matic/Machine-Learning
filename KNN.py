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
#if there are any ties for number of classes in k nearest neighbors, flip a coin
def knn(k, c, data, labels, new_data, distance):
    N1 = np.shape(data)[0]
    N2 = np.shape(new_data)[0]
    new_labels = np.zeros(N2)
    #brute force for now
    for i in range(N2):
        d = np.zeros(N1)
        for j in range(N1):
            d[j] = ss.distance.euclidean(data[j,:], new_data[i,:])
        x = np.argsort(d)[:k] #indices in d, data rows, or labels, for the k nearest neighbors
        classes = np.zeros(c) #the value in labels points back to the index in classes
        for neighbor in x:
            classes[labels[neighbor]] += 1
        new_labels[i] = np.argmax(classes)
    return new_labels


def knn_random_ties(k, c, data, labels, new_data, distance):
    N1 = np.shape(data)[0]
    N2 = np.shape(new_data)[0]
    new_lables = np.zeros(N2)
    #brute force for now
    for i in range(N2):
        d = np.zeros(N1)
        for j in range(N1):
            print distance(data[j:], new_data[i,:])
            d[j] = distance(data[j:], new_data[i,:])
        x = np.argsort(d)[:k] #indices in d, data rows, or labels, for the k nearest neighbors
        classes = np.zeros(c) #the value in labels points back to the index in classes
        for neighbor in x:
            classes[labels[x]] += 1
        x2 = np.argsort(classes)[::-1] #sort the classes from most to least counts
        if classes[x2[0]] > classes[x2[1]]:
            new_labels[i] = x2[0] #if there's no tie, populate
        else:
            x3 = [x2[0]]
            tie = x2[1]
            count = 1
            while c2[count] == x2[0]:
                x3.append(x2[count])
                count += 1
            new_labels[i] = nx3[p.random.randint(len(x3))]
    return new_labels


data = np.zeros((100, 2))
data[:50, 0] = np.random.normal(3, 1, 50)
data[:50, 1] = np.random.normal(3, 1, 50)
data[50:, 0] = np.random.normal(0, 1, 50)
data[50:, 1] = np.random.normal(0, 1, 50)

labels = np.zeros(100)
labels[:50] = 0
labels[50:] = 1

c = 2
k = 10
new_data = np.zeros((100, 2))
new_data[:50, 0] = np.random.normal(3, 1, 50)
new_data[:50, 1] = np.random.normal(3, 1, 50)
new_data[50:, 0] = np.random.normal(0, 1, 50)
new_data[50:, 1] = np.random.normal(0, 1, 50)

my_labels = np.zeros(100)
my_labels[:50] = 0
my_labels[50:] = 1

distance = euclidean_distance

start_time = TIME.time()
new_labels = knn(k, c, data, labels, new_data, distance)
stop_time = TIME.time()
print (stop_time - start_time)

tal = 0.
for i in range(len(my_labels)):
    if my_labels[i] == new_labels[i]:
        tal += 1

tal = tal*100/len(my_labels)
print tal

data = load_iris()

def knn_default(k, c, data, labels, new_data):
    N1 = np.shape(data)[0]
    N2 = np.shape(new_data)[0]
    new_labels = np.zeros(N2)
    #brute force for now
    for i in range(N2):
        d = np.zeros(N1)
        for j in range(N1):
            d[j] = ss.distance.euclidean(data[j,:], new_data[i,:])
        x = np.argsort(d)[:k] #indices in d, data rows, or labels, for the k nearest neighbors
        classes = np.zeros(c) #the value in labels points back to the index in classes
        for neighbor in x:
            classes[labels[neighbor]] += 1
        new_labels[i] = np.argmax(classes)
    return new_labels

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
new_labels2 = knn_default(5, 3, train_data, train_labels, test_data)
new_labels3 = knn_vdist(5, 3, train_data, train_labels, test_data, euclidean_distance)


#to do
# 1) check first implementation and get the function handle for distance to work
# 2) implement on the iris data set and check that it works
# 3) do K-means next
