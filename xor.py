import numpy as np
import sys, os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def unifs(n):
    """
    returns a list of n randomly sampled floats from 0 to 1
    """
    return [np.random.uniform() for i in range(n)]

class NN:
    def __init__(self, ninput, layers=[2,1]):
        self.layers = layers
        self.ninput = ninput
        # There is a +1 here to each of the weights because we introduce the bias as a weight.
        # When we take calculate the next layer, we just add a [1] to the front of the vector to take the dot product.
        edges = [ninput+1] + [i + 1 for i in layers[:-1]]
        print("edges: ", edges)
        self.nodes = [[Node(edges[i]) for j in range(layers[i])] for i in range(len(layers))]
        self.yi_hat = False

    def reweight(self, y):
        # Preamble
        assert self.yi_hat
        yi_hat = self.yi_hat
        assert (len(yi_hat) == len(y))
        layers = self.layers
        nlayers = len(layers)
        yi_hat = self.yi_hat

        # Functionality
        activs = [[node.a for node in layer] for layer in self.nodes]
        loss = rho(y, yi_hat)
        errors = []
        deltaij = 0
        for i in [nlayers - i - 1 for i in range(nlayers)]: # for each layer
            layer_errors = []
            if i == 1:
                delta
                continue

            if i == nlayers - 1:
                continue

            for j in range(layers[i]): # for each node in layer_i

                layer_errors.append(deltaij)

            errors.append(layer_errors)
        print(errors)
        return 0


    def calcNetwork(self, x):
        assert(len(x) == self.ninput)
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
                nodes[i][j].z = z
                print(sig(z))
                nodes[i][j].a = sig(z)
        yhat = [n.a for n in nodes[-1]]
        self.yi_hat = yhat
        return yhat

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
        self.z = 0
        self.a = 0

    def __repr__(self):
        return "Node(" + str(self.a) + ", " + str(self.n) + ")"

    def __str__(self):
        return "Node(" + str(self.t) + ")"



def rho(y, yhat):
    assert(len(y) == len(yhat))
    return sum([(y[i] - yhat[i])**2 for i in range(len(y))])

def sig(x):
    return 1/(1+np.exp(-x))

def sig_prime(x):
    return sig(x) * (1-sig(x))

def main():
    np.random.seed(20885914)
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0,1,1,0]
    netwrk = NN(2)
    blockPrint()
    yhat = netwrk.calcNetwork([0,1])
    enablePrint()
    netwrk.reweight([0])

if __name__ == "__main__":
    main()