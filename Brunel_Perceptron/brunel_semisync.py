###Burnel Perceptron Asynchronous
##perceptron network
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import time as TIME

f = 0.5 #between 0 and .5
N = 200
T = N
K = 0.
max_iter = 100
max_T = 256*T

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

def learn_active_1_step(w, n, t):
    k_i = np.inner(w, n)
    y = threshold_active(k_i, t)
    if y == 1:
        return w, 1
    else:
        w += n
        return w, 0

def learn_inactive_1_step(w, n, t):
    k_i = np.inner(w, n)
    y = threshold_inactive(k_i, t)
    if y == 1:
        return w, 1
    else:
        w -= n
        return rect(w), 0

def network_check(W, n, T, K):
    x = np.inner(W, n)
    c = 0
    for i in range(len(T)):
        if (n[i] == 0) & (x[i] < T - K):
            c+=1
        elif (n[i] == 1) & (x[i] > T + K):
            c+=1
    if c == len(T):
        check = 1
    else:
        check = 0
    return check

#start with random matrix, max_iter = 4000, max_T = 4096N, N = 800, f = 0.5
#new algorithm: start with a single pattern, and train till the whole network gets it
#then start adding



def Brunel_Asynchronous_Learning(N, max_iter, max_T, f, K):
    W = np.random.uniform(0, 10, (N,N))
    failures = [] #all neurons who could not learn all patterns
    p = []
    survivors = np.arange(N) #initial list of neurons
    T = np.zeros(N) + N
    cpat = np.zeros(N)
    training_sessions = 0
    checks = 0
    while (len(survivors) > 0) & (training_sessions < 2*N):
        print "I'm on the ", training_sessions, " th training session with ", len(survivors), " many survivors, and ", len(p), " many patterns"
        #generate pattern
        X = np.random.uniform(0, 1, N)
        n = heaviside(X, f)
        p.append(n)
        WS = []
        WSA = []
        for u in range(len(p)):
            it = 0
            while (it < max_iter) & (checks != 1):
                for i in range(len(survivors)):
                    if p[u][survivors[i]] > f:
                        tk = T[survivors[i]] + K
                        v, check = learn_active_1_step(cp.copy(W[survivors[i],:]), p[u], tk)
                        W[survivors[i],:] = v
                        if check == 1:
                            NP[u, i] = 1
                        else:
                            NP[u,i] = 0
                    if p[u][survivors[i]] < f:
                        tk = T[survivors[i]] - K
                        v, check = learn_inactive_1_step(cp.copy(W[survivors[i],:]), p[u], tk)
                        W[survivors[i],:] = v
                        if check == 1:
                            NP[u,i] += 1
                        else:
                            NP[u,i] = 0
                it += 1
        failures = [survivors[j] for j in range(len(survivors)) if np.sum(NP[:, j]) < len(p)]
        print "FAILURES: ", failures
        for fail in failures:
            T[fail] *= 2.
            W[fail,:] *= 2.
        dunzo = [d for d in range(N) if T[d] >= max_T]
        print "DUNZO: ", dunzo
        survivors = np.sort(list(set(survivors) - set(dunzo)))
        print "SURVIVORS: ", survivors
        training_sessions += 1
    #final check over all patterns and neurons
    for u in range(len(p)):
        for i in range(N):
            if p[u][i] > f:
                tk = T[i] + K
                v, check = learn_active_1_step(cp.copy(W[i,:]), p[u], tk)
                if check == 1:
                    cpat[i] += 1
            if p[u][i] < f:
                tk = T[i] - K
                v, check = learn_inactive_1_step(cp.copy(W[i,:]), p[u], tk)
                if check == 1:
                    cpat[i] += 1
    #final check for patterns that work for the whole network
    pc = []
    for u in range(len(p)):
        x = np.inner(W, p[u])
        c = 0
        for xi in range(N):
            if (p[u][xi] == 1) & (x[xi] > T[xi] + K):
                c+=1
            elif (p[u][xi] == 0) & (x[xi] < T[xi] - K):
                c+=1
        if c == N:
            pc.append(u)

    return W, cpat, p, T, pc

start_time = TIME.time()
W, cpat, p, T, pc = Brunel_Asynchronous_Learning(N, max_iter, max_T, f, K)
stop_time = TIME.time()

print "time: ", stop_time - start_time
print "Whole Network Successful on ", len(pc), " patterns"
ID = np.zeros(N)
OD = np.zeros(N)

for i in range(N):
    idg = len(np.where(W[i,:] > 0)[0])
    ID[i] = idg
    odg = len(np.where(W[:,i] > 0)[0])
    OD[i] = odg

CVID = np.std(ID)/np.mean(ID)
CVOD = np.std(OD)/np.mean(OD)

print "CVID: ", CVID
print "CVOD: ", CVOD

























#