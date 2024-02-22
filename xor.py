import numpy as np

def unifs(n):
    """
    returns a list of n randomly sampled floats from 0 to 1
    """
    return [np.random.uniform() for i in range(n)]

class NN:
    def __init__(self, xi, yi, layers=[2,1]):
        self.layers = layers
        self.xi = xi
        self.yi = yi
        # There is a +1 here to each of the weights because we introduce the bias as a weight.
        # When we take calculate the next layer, we just add a [1] to the front of the vector to take the dot product.
        edges = [len(xi)+1] + [i + 1 for i in layers[:-1]]
        print("edges: ", edges)
        self.nodes = [[Node(edges[i]) for j in range(layers[i])] for i in range(len(layers))]
        self.yi_hat = False

    def reweight(self, yhat, y):
        assert(len(yhat) == len(y))
        loss = rho(y,yhat)
        pass

    def calcNetwork(self, x):
        layers = self.layers
        nodes = self.nodes
        print(nodes)
        for i in range(len(layers)):
            for j in range(layers[i]):
                if i == 0:
                    prev_layer = [1] + x
                else:
                    prev_layer = [1] + [n.a for n in nodes[i-1]]
                print("Prev layer", i, j,"; ", prev_layer)
                print("weights", nodes[i][j].t)
                z = np.dot(prev_layer, nodes[i][j].t)
                print(sig(z))
                nodes[i][j].a = sig(z)
        m = len(layers)
        self.yi_hat = layers[m-1]
        return layers[m-1]

    def last_layer(self):
        return 0

    def rho(self):
        return 0

class Node:
    def __init__(self, n, theta=[]):
        self.n = n
        if theta:
            assert(len(theta) == n)
            self.t = theta
        else:
            self.t = [np.random.uniform() for i in range(n)]
        self.a = 0

    def __repr__(self):
        return "Node(" + str(self.a) + ", " + str(self.n) + ")"

    def __str__(self):
        return "Node(" + str(self.t) + ")"



def rho(y, yhat):
    assert(len(y) == len(yhat))
    return sum([(y[i] - yhat[i])^2 for i in range(len(y))])

def sig(x):
    return 1/(1+np.exp(-x))

def sig_prime(x):
    return sig(x) * (1-sig(x))

def main():
    np.random.seed(20885914)
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0,1,1,0]
    netwrk = NN([0,1],1)
    netwrk.calcNetwork([0,1])

if __name__ == "__main__":
    main()