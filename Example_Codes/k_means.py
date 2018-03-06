import numpy as np
import matplotlib.pyplot as plt
import time as TIME
import scipy.spatial as ss
from sklearn.datasets import load_iris
import math

def euclidean_distance(v1, v2):
    distance = v1 - v2
    ip = np.inner(distance, distance)
    return np.sqrt(ip)

def mean(data):
    n_obs, n_dims = np.shape(data)
    final = np.zeros(n_dims)
    for i in range(n_dims):
        final[i] = np.mean(data[i,:])
    return final

def k_means(K, data, k0):
    N, D = np.shape(data)
    K_Old = np.zeros((K,D))
    K_New = k0
    count = 0
    K_store = [K_Old, K_New]
    Labels = np.zeros(N)
    while (K_store[0] != K_store[1]).all() & (count < 100):
        K_store[0] = K_New
        K_New, Labels = KM_iterate(K_New, data, K, D)
        K_store[1] = K_New
        count +=1
    return K_New, Labels

def distancia(data_point, K, K_current):
    distances = np.zeros(K)
    for i in range(K):
        distances[i] = euclidean_distance(data_point, K_current[i,:])
    return distances
#loop over every data point and calculate distance from current
#dim is number of features
def KM_iterate(K_current, data, K, dim):
    labels = np.zeros(N)
    centroids = np.zeros((K, dim))
    K_counts = np.zeros(K)
    #first assign everyone to the class with the nearest centroid
    for i in range(N):
        distances = distancia(data[i,:], K, K_current) #distance from each centroid
        index = np.argmin(distances) #which centroid is nearest
        centroids[index,:] += data[i,:]
        K_counts[index] +=1
        labels[i] = index
    for k in range(K):
        centroids[k,:] /= K_counts[k]
    return centroids, labels

iris = load_iris()
K = 3
data = iris['data']
N, D = np.shape(data)
k0 = np.zeros((K, D))
k0[0,:] = data[20,:]
k0[1,:] = data[89,:]
k0[2,:] = data[122,:]

centroids, labels = k_means(K, data, k0)
true_centroids = np.zeros((K,D))
for i in range(3):
    mask = np.where(iris.target == i)[0]
    true_centroids[i,:] = mean(iris.data[mask,:])


print(sum(labels == iris.target)/float(N), " is percent accuracy assuming K = 3")

print(centroids, "are the predicted centers of mass\n")
print(true_centroids, "are the real centers of mass")
