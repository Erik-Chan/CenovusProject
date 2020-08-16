import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
import itertools

#tools needed for TSP
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

#Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)

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
        self.edge.append([v1,v2])

    # Remove edges
    def remove_edge(self, v1, v2):
        if [v1,v2] not in self.edge:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.edge.remove([v1,v2])

    def __len__(self):
        return self.size

    def __add__(self, other):
        add_edge(other[0],other[1])
        return self

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

    def __add__(self, other):
        self.add_vertex(other)
        return self

class MSEopt():
    def __init__(self, beta = 0.3, sizeS = 5):
        self.x_sum = 0
        self.beta = beta
        self.sizeS = sizeS
    def eval_obj_fun(self, x):
        self.x_sum = np.sum(x)
        return self.x_sum
        pass
    def set_T(self,T):
        self.T = T
    def set_beta(self, beta):
        self.beta = beta
    def set_lambda_s(self, lmbda):
        self.lmbda = lmbda
    def set_M(self, lambda_s, vertex):
        vertex_perms =  itertools.permutations(S,2)
        m = max([sum(v) for v in list(vertex_perms)])
        #Still need to maximize for all t
        pass
    def get_psi_u(self):
        self.psi_u = np.floor(self.beta*self.T)
        return self.psi_u

toggle = 0
if toggle:
    timeline = np.linspace(0., 20., 10000)
    x = my_ou(x0=1, paths=3, steps=100)(timeline)
    plt.plot(x)
    plt.show()


MSE = 1

if MSE:
    verts = Vertex(vertices = ['Calgary', 'Boston', 'Cushing'])

    edges = Edge(size = 3)
    edges.add_edge(1,2)
    edges.add_edge(1,3)
    edges.add_edge(2,3)
    #costs = {e:np.random.uniform(0,1) for e in edges.get_edge()}
    costs = {tuple(e): np.random.uniform(0,1) for e in edges.get_edge()}

    model = Model()
    T = 1
    t = np.linspace(0,T,10)
    # These are the psi^t
    x = [model.add_var(var_type=BINARY) for _t in t]
    # These are the gamma_s^t
    xx = [[model.add_var(var_type=BINARY) for _t in t] for s in verts.get_vertex()]

    #These are the alpha_s
    alpha_s = [model.add_var() for s in verts.get_vertex()]


    #model += alpha_s[1] >= 500
    model.objective = minimize(sum(alpha_s[i] for i in set(range(len(verts)))))
    model.optimize()
    print('the solution at the minimum is :', alpha_s[0].x, alpha_s[1].x, alpha_s[2].x)
