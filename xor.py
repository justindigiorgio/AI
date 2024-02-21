import numpy as np



class NN:
    def __init__(self, x, y, layers=[2,1]):
        self.n = len(layers)
        self.x = x + [1]
        self.y = y
        edges = [len(x) + 1] + [layers[i] + 1 for i in range(len(layers)-1)] + [layers[-1]]
        print(edges)
        self.layers = [[Node(edges[i]) for j in range(edges[i+1])] for i in range(len(edges) - 1)]
        print([[j.t for j in i] for i in self.layers])
        self.yhat = NN.calcNetwork(self)

    def calcNetwork(self):
        layer = 0
        n = self.n
        i = 0
        while i < n:
            nodes = self.layers[i]
            if i == 0:
                prev_layer = self.x
            else:
                prev_layer = self.layers[i-1]

        return 0

    def last_layer(self):
        return 0

    def rho(self):
        return 0


class Node:
    def __init__(self, n, theta=[]):
        if theta:
            assert(len(theta) == n)
            self.t = theta
        else:
            self.t = [np.random.uniform() for i in range(n)]
        self.a = 0

    def activ(self, prev_layer):
        assert(len(prev_layer) == len(self.t))
        activ = sig(np.dot(prev_layer, self.t))
        self.a = activ
        return activ

def rho():
    return 0

def sig(x):
    return 1/(1+np.exp(-x))

def sig_prime(x):
    return sig(x) * (1-sig(x))

def main():
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0,1,1,0]
    netwrk = NN(x,y)

if __name__ == "__main__":
    main()