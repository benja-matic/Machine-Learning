import numpy as np
import copy as cp
import matplotlib.pyplot as plt
np.random.seed(1234)

def sum_square_error(hx, y):
    return np.sum((hx - y)**2.)

def abs_error(hx, y):
    return np.sum(np.abs(hx-y))

def linear_hypothesis(m, x, b):
    return (m*x)+b

def gradient(beta, x, y):
    return np.sum(x*((beta*x)-y))

x = np.arange(100)
true_slope = 3.456789
y = x*true_slope

theta_0 = 87.
theta = theta_0 + 0.
alpha = .0000005

errors = np.zeros(200)

for i in range(200):
    e = sum_square_error(theta*x, y)
    gradiente = gradient(theta, x, y)
    errors[i] = e
    theta = theta - alpha*gradiente



print("Initial Guess: {0}".format(theta_0))
print("Final Guess: {0}".format(theta))
print("True Beta: {0}".format(true_slope))

plt.plot(errors)
plt.xlabel("Iteration #")
plt.ylabel("Sum Square Error")
