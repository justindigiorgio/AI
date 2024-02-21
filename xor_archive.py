import numpy as np
import random


def NN(theta, xi):
    """
    Applies a 2/2/1 neural network to a starting value x given a theta.
    """
    print("NN Theta: ", theta)
    print("NN X: ", xi)
    xi.append(1)
    z1 = np.dot(theta[0], xi)
    z2 = np.dot(theta[1], xi)
    z3 = np.dot(theta[2], [sig(z1),sig(z2),1])
    print("Output value: ",z3)
    return z3


def rand_thetas():
    theta = [[np.random.uniform() for i in range(3)] for j in range(3)]
    print("Random theta:", theta)
    return theta


def rho(thetas, x, y):
    print("Rho Call")
    print("Thetas: ", thetas)
    print("X: ", x)
    print("Y: ", y)
    r = [(yi - NN(thetas,xi))^2 for xi in x for yi in y]
    return sum(r)


def backprop(loss, theta):



def grad(thetas, x,y):
    print("Gradient Call")
    print("Thetas: ", thetas)
    print("X: ", x)
    print("Y: ", y)
    r = [y[i] - NN(thetas, x[i]) for i in range(3)]
    g1 = [ri * xi[0] for ri in r for xi in x]
    g2 = [ri * xi[1] for ri in r for xi in x]
    return [-1 * sum(g1), -1 * sum(g2), -1 * sum(r)]

def sig(x):
    return 1/(1+np.exp(-x))

def siginv(y):
    return -np.log(1/y - 1)

def lineSearch(theta, g, step=0.01, stepMax=0.5):
    print("Line Search: ", theta, g)
    mag = np.sqrt(sum([gi**2 for gi in g]))
    dir = [gi / mag for gi in g]
    line =
    sig_thetas = [sig(t) for t in theta]
    sig_theta_hats = [l + s for l in line for s in sig_thetas]
    theta_hats = [siginv(s) for s in sig_theta_hats]
    losses = [rho(t) for t in theta_hats ]
    ind = np.argmin(losses)
    return theta_hats[ind]

def checkConvergence(t1, t2, tol=0.01, relative=False):
    d1 = sum([t1i**2 for t1i in t1])**0.5
    d2 = sum([t2i ** 2 for t2i in t2]) ** 0.5
    if relative:
        return abs(d2/d1 - 1) < tol
    else:
        return abs(d2-d1) < tol

def grad_descent(theta0,x,y, maxi=50):
    theta_i = theta0
    i = 0
    converged = False
    while not converged and i < maxi:
        g = grad(theta0, x, y)
        theta_new = lineSearch(theta_i, g)
        converged = checkConvergence(theta_i, theta_new)
        theta_new = theta_i
        i += 1
    return theta_i

def main():
    np.random.seed(20885914)
    data = [[[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]]
    theta0 = rand_thetas()
    w_opt = grad_descent(theta0, data[0], data[1])
    print(w_opt)

if __name__=="__main__":
    main()