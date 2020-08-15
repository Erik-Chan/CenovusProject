import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
import itertools
#Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################


@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k*(theta - x), 'dw': sigma}

class Edge():

    # Initialize the edge set
    def __init__(self, size = 0):
        self.edge = []
        self.edge = [[i,i] for i in range(0,size)]
        self.size = size

    # Add edges
    def add_edge(self, v1,v2):
        if [v1, v2] in self.edge:
            print("Edge from %d to %d already exists" % (v1, v2))
            return
        self.adjMatrix.append([v1,v2])

    # Remove edges
    def remove_edge(self, v1, v2):
        if [v1,v2] not in self.edge:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix.remove([v1,v2])

    def __len__(self):
        return self.size

    def get_edge(self):
        return self.edge

class Vertex():
    # Initialize the vertex set
    def __init__(self, vertices = []):
        self.vertex = []
        self.vertex = [x for x in vertices]

    def add_vertex(self, v1):
        if v1 in self.vertex:
            print("Vertex %d already exists" % v1)
            return
        self.vertex.append(v1)

    def remove_vertex(self, v1):
        if v1 not in self.vertex:
            print("No vertex labelled %d exists" % v1)
            return
        self.vertex.remove(v1)

    def __len__(self):
        return len(self.vertex)

    def get_vertex(self):
        return self.vertex

class MSEopt():
    def __init__(self, beta = 0.3, sizeS = 5):
        self.x_sum = 0
        self.beta = beta
        self.sizeS = sizeS
    def eval_obj_fun(self, x):
        self.x_sum = np.sum(x)
        return self.x_sum
        pass
    def set_beta(self, beta):
        self.beta = beta
    def set_lambda_s(self, lmbda):
        self.lmbda = lmbda
    def set_M(self, lambda_s, vertex):
        vertex_perms =  itertools.permutations(S,2)
        m = max([sum(v) for v in list(vertex_perms)])
        #Still need to maximize for all t
        pass







toggle = 0
if toggle:
    timeline = np.linspace(0., 20., 10000)
    x = my_ou(x0=1, paths=3, steps=100)(timeline)
    plt.plot(x)
    plt.show()