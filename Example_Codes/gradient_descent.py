import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

def batch_gradient_V(theta, X, y, alpha, m):
    m1 = 1./m
    m2 = 1./(2.*m)
    Xt = np.transpose(X)
    hypothesis = np.inner(theta, X)
    loss = hypothesis-y
    J = np.sum(loss**2.)*m2
    gradient = np.inner(Xt, loss)/m
    theta -= alpha * gradient
    return theta, J, gradient


m = 100 #number of training examples
x_axis = np.arange(m)
true_slope = 8.456789
true_intercept = 7.
X = np.zeros((m, 2))
y = (x_axis*true_slope) + true_intercept + np.random.normal(0, 2, m)

X[:,0] = 1.
X[:,1] = x_axis

alpha = .0001
theta = np.array([2., 17.])
iterations = 100000
error = np.zeros(iterations)
grads = np.zeros((2, iterations))
grits = np.zeros((2, iterations))

for i in range(iterations):
    theta, e, g = batch_gradient_V(theta, X, y, alpha, m)
    error[i] = e
    grads[:,i] = g


plt.plot(X[:,1], y, "g.", ms = 10.)
plt.plot(X[:,1], np.inner(theta, X), "r")
plt.show()
#to do: implement a gradient checker using J(theta + epsilon)
