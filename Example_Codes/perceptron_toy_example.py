###Coding (Single-Layer) Perceptron From Scratch
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
#y = f(z) maps input vector z to 1 or 0
#D = {(x1, d1),...,(xs, ds)} is a training set of vectors and associated decisions
#W = weights vector element-wise multiplied by feature vector z
#x_i,j = value of ith FEATURE of jth INPUT VECTOR
#j is the vector, i is element of vector
#x_j,0 = 0
#w_i(t) = weight i at time t

print "This example creates cases and controls in a 2D plane where a line with a slope of 1 separates them\n\
Then we use the perceptron algorithm to classify test data"

dim = 2
t = 1
n_samples = 1000
n_retest = 1000

#y must be greater than x
#ground truth
def check_case(x):
    if x[1] > x[0]:
        return 1
    else:
        return 0

def threshold(x):
    if x > 0:
        return 1
    else:
        return 0

d_train = np.random.normal(0.5, 3, (dim, n_samples)) #generate training data
c_train = [check_case(d_train[:, i]) for i in range(n_samples)] #classify training data

W = np.zeros(dim)

d_test = np.random.normal(0.5, 3, (dim, n_retest)) #generate test data
c_test = [check_case(d_test[:, i]) for i in range(n_retest)] #classify test data

#train classifier on one set of n_samples
for i in range(n_samples):
    y = threshold(np.inner(W,d_train[:,i])) #inner product returns a scalar, threshold checks if scalar above or below 0
    for j in range(dim):
        W[j] += (c_train[i] - y)*d_train[j, i] #(correct classification - perceptron output)*training vector[j]



#test classifier
correct = []
incorrect = []
fig, ax = plt.subplots(3,1)

for i in range(n_retest):
    y = threshold(np.inner(W, d_test[:,i]))
    if y == 1:
        ax[1].plot(d_test[0,i], d_test[1,i], "g.")
    else:
        ax[1].plot(d_test[0,i], d_test[1,i], "r.")
    c = c_test[i]
    if y == c:
        correct.append(i)
    else:
        incorrect.append(i)

ax[1].set_title("Perceptron Output on Test Data")

accuracy = float(len(correct))/n_retest
print accuracy, "is percentage correct on test set"

for i in range(n_samples):
    if c_test[i] == 1:
        ax[0].plot(d_test[0, i], d_test[1, i], "g.")
    else:
        ax[0].plot(d_test[0, i], d_test[1, i], "r.")

ax[0].set_title("Ground Truth Test Data")

d_c = d_test[:, correct]
d_i = d_test[:, incorrect]

ax[2].plot(d_c[0,:], d_c[1,:], "k.")
ax[2].plot(d_i[0,:], d_i[1,:], "c.", ms = 3)

ax[2].set_title("Perceptron Errors in Cyan on Test Data")

ax[2].plot([np.min(d_test[0,:]), np.max(d_test[0,:])], [np.min(d_test[0,:]), np.max(d_test[0,:])], "r", linestyle = 'dashed')



for i in range(3):
    ax[i].set_xticks([])


# for i in range(len(correct)):
#     ax[2].plot(d_test[0, correct[i]], d_test[1, correct[i]], "k.")
#
# for i in range(len(incorrect)):
#     ax[2].plot(d_test[0, incorrect[i]], d_test[1, incorrect[i]], "c", lw = 5)
#

plt.tight_layout()
plt.show()
