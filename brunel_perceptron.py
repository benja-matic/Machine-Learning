##perceptron network
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import time as TIME

f = 0.5 #between 0 and .5
N = 1000
T = N
K = 0.

def heaviside(x, t):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > t:
            y[i] = 1
        else:
            y[i] = 0
    return y

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

def learn_active_1_step(w, n, t):
    k_i = np.inner(w, n)
    y = threshold_active(k_i, t)
    w += (1. - y)*n
    return w

def learn_inactive_1_step(w, n, t):
    k_i = np.inner(w, n)
    y = threshold_active(k_i, t)
    w += (1. - y)*(-n)
    return w







#given an initial matrix, max_iter = 4000, max_T = 4096*N, N = 800
#while we have relevant neurons
#generate patterns on the fly, and for all relevant neurons, learn the pattern
#update the weights matrix with the best attempt at learning
#if you're on iteration > 1, now run through the set of ps and check them all
#if you can't learn, T*=2 for you, W[i,:] *= 2
#if your T >= max_T, you become irrelevant
#






#all neurons relevant, no failures, p = [], c = 0
#while at least 1 relevant neuron
#draw random pattern, add to p, cpat = zeros
#each neuron tries to learn all patterns up to this point (do we need to train asynchronously?)
#cpat[neuron] +=1 for all learned patterns
#failures.append(i) for all failed patterns
#for all failures, update thresholds
#recreate survivor list subtracting out losers
#
def Brunel_Learning(W, max_iter, max_T, N):
    conditions = False
    failures = [] #all neurons who could not learn all patterns
    p = []
    survivors = np.arange(N) #initial list of neurons
    T = np.zeros(N) + N
    cpat = np.zeros(N)
    c = 0
    while len(survivors) > 0:
        #generate pattern
        X = np.random.uniform(0, 1, N)
        n = heaviside(X, f)
        p.append(n)
        #have every neuron learn the pattern
        #survivors contains the list of all neurons T < 4096N
        for u in p:
            for i in range(len(survivors)):
                if n[survivors[i]] > f:
                    tk = T[survivors[i]] + K
                    v, itr = learn_active_1v(cp.copy(W[i,:]), u, tk, max_iter)
                    if itr == max_iter:
                        failures.append(survivors[i]) #ensure that failures contains correct neuron #
                    else:
                        cpat[survivors[i]] += 1
                    W[survivors[i,:]] = v
                if n[survivors[i]] < f:
                    tk = T[survivors[i]] - K
                    v, itr = learn_inactive_1v(cp.copy(W[i,:]), u, tk, max_iter)
                    if itr == max_iter:
                        failures.append(survivors[i])
                    else:
                        cpat[survivors[i]] += 1
                    W[survivors[i,:]] = v
        for fails in failures:
            T[i] *= 2.
            W[i,:] *= 2.
        dunzo = [i for i in range(N) if T[i] >= max_T]
        survivors = np.sort(list(set(survivors) - set(dunzo)))

    return W, cpat


def Brunel_Learning(W, max_iter, max_T, N):
    conditions = False
    failures = [] #all neurons who could not learn all patterns
    p = []
    survivors = np.arange(N) #initial list of neurons
    T = np.zeros(N) + N
    cpat = np.zeros(N)
    c = 0
    while len(survivors) > 0:
        #generate pattern
        X = np.random.uniform(0, 1, N)
        n = heaviside(X, f)
        p.append(n)
        #have every neuron learn the pattern
        #survivors contains the list of all neurons T < 4096N
        for u in p:
            for i in range(len(survivors)):
                if n[survivors[i]] > f:
                    tk = T[survivors[i]] + K
                    v, itr = learn_active_1v(cp.copy(W[i,:]), u, tk, max_iter)
                    if itr == max_iter:
                        failures.append(survivors[i]) #ensure that failures contains correct neuron #
                    else:
                        cpat[survivors[i]] += 1
                    W[survivors[i,:]] = v
                if n[survivors[i]] < f:
                    tk = T[survivors[i]] - K
                    v, itr = learn_inactive_1v(cp.copy(W[i,:]), u, tk, max_iter)
                    if itr == max_iter:
                        failures.append(survivors[i])
                    else:
                        cpat[survivors[i]] += 1
                    W[survivors[i,:]] = v
        for fails in failures:
            T[i] *= 2.
            W[i,:] *= 2.
        dunzo = [i for i in range(N) if T[i] >= max_T]
        survivors = np.sort(list(set(survivors) - set(dunzo)))

    return W, cpat


W0 = np.random.uniform(0, 5, (N,N))
'''testing codes

W = np.random.uniform(0, 10, (N,N))
p = []
u1 = heaviside(np.random.uniform(0, 1, N), f)
u2 = heaviside(np.random.uniform(0, 1, N), f)
p.append(u1)
p.append(u2)
it = 0
NPS = 2*N
NP = np.zeros((2, N))
while (it < 4000) & (np.sum(NP) < NPS):
    for u in range(2):
        for i in range(N):
            if p[u][i] > 0:
                v, check = learn_active_1_step(cp.copy(W[i,:]), p[u], tk)
                W[i,:] = v
                if check == 1:
                    NP[u, i] = 1
                else:
                    NP[u, i] = 0
            else:
                v, check = learn_inactive_1_step(cp.copy(W[i,:]), p[u], tk)
                W[i,:] = v
                if check == 1:
                    NP[u, i] = 1
                else:
                    NP[u, i] = 0
    it += 1


'''








        #
