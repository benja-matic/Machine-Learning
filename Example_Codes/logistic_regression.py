###Coding (Single-Layer) Perceptron From Scratch
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def linear_function(W, X, b):
    """W should have dimensions nx1
    X should have dimensions nxm (parameters x training examples)
    b is just a scalar that gets broadcast
    """
    return np.dot(W.T, X) + b


def logistic_loss(Y, A):
    """
    y is a 1xm vector of training examples, coded as 1 or 0
    a is the output of the sigmoid function operating on W.T * X + b, so is also 1xm
    """
    m = Y.shape[1]# number of training examples
    loss_vector = -(Y*np.log(A) + (1. - Y)*np.log(1.-A))# An m - array with the loss for each training example
    return np.sum(loss_vector) / m# Cost function is sum over the losses / training examples

#you have a 1xm vector, want da/dz for all m training examples
def d_sigmoid_d_z(z):
    s = sigmoid(z)
    return s*(1. - s)


def d_L_d_a(Y, A):
    return -((Y/A) + (1. - Y)/(1. - A))


def d_L_d_z(Y, A):
    return A - Y

###dL/dz = a - y
###dz/dw = x
###dz/db = 1

def d_L_d_w(A, Y, X):
    return d_loss_d_z(Y, A)*X


def check_case(x):
    if x[1] > x[0]:
        return 1
    else:
        return 0


def gradient_descent_logistic_regression(X, Y, max_iter, tolerance, alpha):
    n, m = X.shape
    W = np.zeros((n,1))#np.random.randn(n,1)
    b = 0
    c = 0
    loss = np.inf
    while (c < max_iter) & (loss > tolerance):
        if c % 200 == 0:
            print("We're on training exmple ", c)
            print("The current loss is ", loss)
        Z = np.dot(W.T, X) + b
        A = sigmoid(Z)
        dZ = A - Y
        dW = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m#np.mean(dZ)#maybe revert to np.sum(dZ) / m?
        W -= alpha*dW
        b -= alpha*db
        c+=1
        loss = logistic_loss(Y, A)
    return W, b, loss, c

n, m = 2, 1000
X = np.random.normal(0.5, 3, (n, m)) #generate training data
Y = np.array([check_case(X[:, i]) for i in range(m)]).reshape(1, m) #classify training data

max_iter = 10000
tolerance = 0.001
alpha = 0.01

W, b, loss, c = gradient_descent_logistic_regression(X, Y, max_iter, tolerance, alpha)

y1 = np.where(Y[0] == 1)[0]
y0 = np.where(Y[0] == 0)[0]

Z_train = np.dot(W.T, X) + b
A_train = sigmoid(Z_train)
a1 = np.where(A_train[0,:] > 0.5)[0]
a0 = np.where(A_train[0,:] <= 0.5)[0]

X_test = np.random.normal(0.5, 3, (n,m))
Y_test = np.array([check_case(X_test[:, i]) for i in range(m)]).reshape(1, m)


Y_1 = np.where(Y_test[0] == 1)[0]
Y_0 = np.where(Y_test[0] == 0)[0]

Z_test = np.dot(W.T, X_test) + b
A_test = sigmoid(Z_test)

A_pred = np.array(A_test > 0.5, dtype = int)
A_diff = np.array(A_pred == Y_test, dtype = int)[0,:]
A_good = np.where(A_diff > 0)[0]
A_errs = np.where(A_diff == 0)[0]

A_1 = np.where(A_test[0,:] > 0.5)[0]
A_0 = np.where(A_test[0,:] <= 0.5)[0]
loss_test = logistic_loss(Y_test, A_test)

fig, ax = plt.subplots(2,1)

ax[0].set_title("Ground Truth")
ax[0].plot(X_test[0, Y_1], X_test[1, Y_1], "g.", label = "True Case")
ax[0].plot(X_test[0, Y_0], X_test[1, Y_0], "r.", label = "True Control")
ax[0].plot(X_test[0, A_errs], X_test[1, A_errs], "k.", label = "Wrong Predictions")
ax[0].legend()

ax[1].set_title("Logistic Regression Predictions")
ax[1].plot(X_test[0, A_1], X_test[1, A_1], "g.")
ax[1].plot(X_test[0, A_0], X_test[1, A_0], "r.")

plt.tight_layout()
plt.show()


# correct = []
# incorrect = []
# fig, ax = plt.subplots(3,1)
#
# for i in range(m):
#     y = threshold(np.inner(W, d_test[:,i]))
#     if y == 1:
#         ax[1].plot(d_test[0,i], d_test[1,i], "g.")
#     else:
#         ax[1].plot(d_test[0,i], d_test[1,i], "r.")
#     c = c_test[i]
#     if y == c:
#         correct.append(i)
#     else:
#         incorrect.append(i)
#



#
