import numpy as np
import copy as cp
import matplotlib.pyplot as plt
np.random.seed(1234)

#Error function takes two np arrays
def mean_square_error(x1, x2):
    return np.mean((x1 - x2)**2.)

def sum_square_error(x1, x2, sigma):
    x3 = sum((x1 - x2)**2.)
    return x3/(2*sigma**2.)


#generate fake data
#assume a model and choose random initial guess for all parameters
#compute chi2_current
#try new parameters and compute chi2_proposed
#compute likelihoods from the error functions
#compute likelihood ratio (exp(-chi2_proposed)/exp(-chi2_current))
#draw a random number 0 < r < 1
#if ratio > r, keep proposed parameters
#store likelihood for each parameter set, and the parameters
#choose parameters with highest likelihood and test
print ("This simulation generates data following a cubic polynomial with added noise\n\
We assume a cubic polynomial model, and use an MCMC search to find the best parameters")

def true_model(x):
    return .2*(x**3.) + .5*(x**2.) -3*x + 14

def test_model(p, x):
    return p[0]*(x**3.) + p[1]*x**2. + p[2]*x + p[3]

noise = np.random.normal(0, 2.5, 1000)
x = np.linspace(-10, 10, 1000)
D = true_model(x) + noise
sigma = 1
a = 2.

iterations = 10000
params = np.zeros((4, iterations))
likeli = np.zeros(iterations)
pc = np.random.normal(0, 1, 4)
Temp = 3.

for i in range(iterations):
    yc = test_model(pc, x)
    chi2_current = sum_square_error(yc, D, sigma)
    pp = pc + np.random.normal(0, Temp, 4)
    yp = test_model(pp, x)
    chi2_proposed = sum_square_error(yp, D, sigma)
    params[:, i] = pc
    likeli[i] = chi2_current
    r = np.random.uniform(0, 1)
    ratio = np.exp(-chi2_proposed + chi2_current)
    if ratio > r:
        pc = pp

d = np.where(likeli == np.min(likeli))[0]
p = params[:, d[0]]

yn = test_model(p, x)
fig, ax = plt.subplots(2,1)
ax[0].plot(x, D, "g", label = "data")
ax[0].plot(x, yn, "r", label = "model")
ax[0].set_title("Best Model Fit to Data")
ax[0].legend()
ax[1].plot(np.log(likeli), "k")
ax[1].set_title("History of Chi Square Values")
ax[1].set_ylabel("Log Chi Square")
plt.tight_layout()
plt.show()
