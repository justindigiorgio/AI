import numpy as np
import random


def NN(theta, xi):
    b = theta[3]
    w = [theta[1], theta[2]]
    z1 = xi[1]*w[1][1] + xi[2]*w[1][2] + b[1]
    z2 = xi[1] * w[2][1] + xi[2] * w[2][2] + b[2]
    z3 = z1 * w[3][1] + z2 * w[3][2] + b[3]
    return z3


def rand_weights(seed=20885914):
    np.random.seed(seed)
    weights = [[np.random.uniform(), np.random.uniform()],
               [np.random.uniform(),np.random.uniform()],
               [np.random.uniform(),np.random.uniform()]]
    return weights

def rand_biases(seed=20885914):
    np.random.seed(seed)
    biases = [np.random.uniform(),np.random.uniform(), np.random.uniform()]
    return biases

def rho(theta, x, y):
    r = [(yi - NN(theta,xi))^2 for xi in x for yi in y]
    return sum(r)

def grad(theta, x,y):
    r = [yi - NN(theta, xi) for xi in x for yi in y]
    g1 = [ri * xi[1] for ri in r for xi in x]
    g2 = [ri * xi[2] for ri in r for xi in x]
    return [-1 * sum(g1), -1 * sum(g2), -1 * sum(r)]

def sig(x):
    return 1/(1+np.exp(-x))

def siginv(y):
    return -np.log(1/y - 1)


def lineSearch(theta, g, step=0.01, stepMax=0.5):
    dir = [gi / (g[1]^2 + g[2]^2 + g[3]^2)^0.5 for gi in g]
    line = [[t[1] + d1 * i, t[2] + d2 * i, t[3] + d3 * i]
            for i in range(0, stepMax, step)
            for d1,d2, d3 in dir
            for t in theta]
    sig_thetas = [sig(t) for t in theta]
    sig_theta_hats = [l + s for l in line for s in sig_thetas]
    theta_hats = [siginv(s) for s in sig_theta_hats]
    losses = [rho(t) for t in theta_hats ]
    ind = np.argmin(losses)
    return theta_hats[ind]

def converged(t1, t2, tol = 0.01, relative=False):
    d1 = sum([t1i^2 for t1i in t1])^0.5
    d2 = sum([t2i ^ 2 for t2i in t2]) ^ 0.5
    if relative:
        return abs(d2/d1 - 1) < tol

def grad_descent(theta0,x,y, maxi=50):
    theta_i = theta0
    i = 0
    converged = False
    while converged and i < maxi:
        g = grad(theta0, x, y)
        theta_new = lineSearch(theta_i, g)

        i += 1

def main():
    data = [[[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]]
    w = rand_weights()
    b = rand_biases()
    print(w)

if __name__=="__main__":
    main()