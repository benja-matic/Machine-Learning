##perceptron network
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import time as TIME

f = 0.5 #between 0 and .5
N = 1000

def heaviside(x, t):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > t:
            y[i] = 1
        else:
            y[i] = 0
    return y

X = np.random.uniform(0, 1, N)
n = heaviside(X, f)

W = np.random.uniform(0,1 , (N,N))

#Thresholding function for perceptron algorithm
def threshold_active(x, t):
        if x > t:
            return 1
        else:
            return 0

def threshold_inactive(x, t):
    if x < t:
        return 1
    else:
        return 0


def rect(x):
    a = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 0:
            a[i] = x[i]
        else:
            a[i] = 0
    return a


#activity (1 or 0), initial weights w, input vector n
#t = T + K for active neurons and T = T - K for inactive neurons
#Does not take k as an argument as it only modifies threshold T, thus pass in T+K or T-K
def perceptron_learning_1v(activity, w, n, t, max_iter):
    i = 0
    won = np.ones(len(n))
    y = 0
    while (y < 1) & (i < max_iter):
        k_i = np.inner(w, n) #input to this neuron
        y = threshold(k_i, t, activity) #is it greater than T = T + K?
        u = np.multiply((won - y), n) #update weights with
        w += u
        i += 1
        print("update by: ", u, "\n")
        print(w, "\n")
    return w, i

def learn_active_1v(w, n, t, max_iter):
    i = 0
    won = np.ones(len(n))
    y = 0
    while (y < 1) & (i < max_iter):
        k_i = np.inner(w, n) #input to this neuron
        y = threshold_active(k_i, t) #is it greater than T = T + K?
        w += np.multiply((won - y), n)
        i += 1
    return w, i

def learn_inactive_1v(w, n, t, max_iter):
    i = 0
    won = np.ones(len(n))
    y = 0
    while (y < 1) & (i < max_iter):
        k_i = np.inner(w, n) #sum of weights times activity states
        y = threshold_inactive(k_i, t) #if you're below threshold, return 1, else, return 0
        u = np.multiply((won - y), -n) #(correct - incorrect)*n
        w += u
        w = rect(w)
        i += 1
    return w, i

def learn_active_1v(w, n, t, max_iter):
    nm = np.where(n > 0)[0]
    i = 0
    won = np.ones(len(n))
    y = 0
    while (y < 1) & (i < max_iter):
        k_i = np.inner(w, n) #input to this neuron
        y = threshold_active(k_i, t) #is it greater than T = T + K?
        w[nm] += 1
        i += 1
    return w, i

def learn_inactive_1v(w, n, t, max_iter):
    nm = np.where(n > 0)[0]
    i = 0
    won = np.ones(len(n))
    y = 0
    while (y < 1) & (i < max_iter):
        k_i = np.inner(w, n) #sum of weights times activity states
        y = threshold_inactive(k_i, t) #if you're below threshold, return 1, else, return 0
        w[nm] -= 1
        w = rect(w)
        i += 1
    return w, i


T = N
K = 10.
print("We have a network of N binary excitatory neurons.\n\
We want to train the network to learn an attractor state eta (written 'n')\n\
The attractor state is a bit string of length N\n\
Learning this attractor state is equivalent to N perceptron probems\n\
Just have to train each neuron to correctly be a 1 or 0\n\
Inputs to a neuron are a sum over incoming synapses times activity levels of those neurons (1 or 0)\n\
Weights matrix is directed and generally asymmetric. Incoming connections are given by one row\n\
row_i*eta --> eta_i for all i in network is a learned attractor state \n\
This simulation learns a single attractor state\n\
When a neuron fails, you increase the threshold and keep learning.\n\
You put a ceiling on iterations or threshold height.")

start_time = TIME.time()
W2 = np.zeros((N, N))

for i in range(len(n)):
    if n[i] > f:
        tk = T + K
        x = learn_active_1v(cp.copy(W[i,:]), n, tk, 4000)[0]
        W2[i,:] = x
    else:
        tk = T - K
        x = learn_inactive_1v(cp.copy(W[i,:]), n, tk, 4000)[0]
        W2[i,:] = x



stop_time = TIME.time()
print(stop_time - start_time, "is time spent learning")

checks = np.zeros(len(n))
vals = np.zeros(len(n))
for i in range(len(n)):
    innie = np.inner(W2[i,:], n)
    vals[i] = innie
    if n[i] > f:
        if innie > T+K:
            checks[i] = 1
    elif n[i] < f:
        if innie < T+K:
            checks[i] = 1


print(100.*sum(checks)/N, "is precent accuracy")
print("here's the weights matrix")

a = plt.imshow(W2)
plt.colorbar(a)
plt.show()
#












        #
