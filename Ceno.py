import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
import itertools

# tools needed for TSP
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)


@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k * (theta - x), 'dw': sigma}


class Edge:

    # Initialize the edge set
    def __init__(self, size=0):
        self.edge = []
        self.edge = [[i, i] for i in range(0, size)]
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if [v1, v2] in self.edge:
            print("Edge from %d to %d already exists" % (v1, v2))
            return
        self.edge.append([v1, v2])

    # Remove edges
    def remove_edge(self, v1, v2):
        if [v1, v2] not in self.edge:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.edge.remove([v1, v2])

    def __len__(self):
        return self.size

    def __add__(self, other):
        add_edge(other[0], other[1])
        return self

    def get_edge(self):
        return self.edge


class Vertex:
    # Initialize the vertex set
    def __init__(self, vertices=None):
        if vertices is None:
            vertices = []
        self.vertex = []
        self.vertex = [v for v in vertices]

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


class MSEopt:
    def __init__(self, beta=0.3, sizeS=5, T=1):
        self.x_sum = 0
        self.beta = beta
        self.sizeS = sizeS
        self.T = T

    def eval_obj_fun(self, obj_v):
        self.obj_v_sum = np.sum(obj_v)
        return self.obj_v__sum
        pass

    def set_T(self, T):
        assert isinstance(T, float)
        self.T = T

    def set_beta(self, beta):
        self.beta = beta

    def set_lambda_s(self, lmbda):
        self.lmbda = lmbda

    def set_M(self, lambda_s, vertex):
        vertex_perms = itertools.permutations(S, 2)
        m = max([sum(v) for v in list(vertex_perms)])
        # Still need to maximize for all t
        pass

    def get_psi_u(self):
        self.psi_u = np.floor(self.beta * self.T)
        return self.psi_u


toggle = 0
plot_toggle = 0
if toggle:
    timeline = np.linspace(0., 1., 10)
    # x = my_ou(x0=1, paths=3, steps=100)(timeline)
    x = my_ou(x0=30, k=2, theta=30, sigma=5, paths=3, steps=10)(timeline)
    print([row[0] for row in x])
    if plot_toggle:
        plt.plot(x)
        plt.show()

MSE = 1

if MSE:
    # Time
    T = 1
    t = np.linspace(0, T, 10)
    # The set \mathcal{S}
    verts = Vertex(vertices=['Calgary', 'Boston', 'Cushing'])

    edges = Edge(size=3)
    edges.add_edge(1, 2)
    edges.add_edge(1, 3)
    edges.add_edge(2, 3)
    # costs = {e:np.random.uniform(0,1) for e in edges.get_edge()}
    costs = {tuple(e): np.random.uniform(0, 1) for e in edges.get_edge()}

    # Note that this should eventually look like [[something for _t in t] for s in verts.get_vertex()]
    # This current version just populates placeholder values.

    # This is the \lambda_{s}^{t}
    localPrices = my_ou(x0=30, k=2, theta=30, sigma=5, paths=3, steps=10)(t)

    model = Model()

    # z = m.add_var(name='zCost', var_type=INTEGER, lb=-10, ub=10)

    # These are the psi^t
    psi_t = [model.add_var(var_type=BINARY) for _t in t]
    # These are the gamma_s^t
    gamma_st = [[model.add_var(var_type=BINARY) for _t in t] for s in verts.get_vertex()]

    # These are the alpha_s
    alpha_s = [model.add_var() for s in verts.get_vertex()]

    # These are the eta^t and independent of s
    eta_t = [model.add_var() for _t in t]

    # These are the rho_s
    rho_s = [model.add_var() for s in verts.get_vertex()]

    # These are \bar{w}_{s}^{t}.
    til_W_st = [[model.add_var() for _t in t] for s in verts.get_vertex()]

    # model += alpha_s[1] >= 500
    model.objective = minimize(xsum(alpha_s[i] for i in set(range(len(verts)))))
    model.optimize()
    print('The solution at the minimum is :', alpha_s[0].x, alpha_s[1].x, alpha_s[2].x)

TSP = 0
if TSP:
    # names of places to visit
    places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
              'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
              'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
              'Remouchamps', 'Waterloo']

    # distances in an upper triangular matrix
    dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
             [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
             [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
             [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
             [51, 114, 72, 54, 69, 139, 105, 155, 62],
             [70, 25, 22, 52, 90, 56, 105, 16],
             [45, 61, 111, 36, 61, 57, 70],
             [23, 71, 67, 48, 85, 29],
             [74, 89, 69, 107, 36],
             [117, 65, 125, 43],
             [54, 22, 84],
             [60, 44],
             [97],
             []]

    # number of nodes and list of vertices
    n, V = len(dists), set(range(len(dists)))

    # distances matrix
    c = [[0 if i == j else dists[i][j - i - 1] if j > i
    else dists[j][i - j - 1] for j in V] for i in V]

    model = Model()

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

    # continuous variable to prevent subtours: each city will have a
    # different sequential id in the planned route except the first one
    y = [model.add_var() for i in V]

    # objective function: minimize the distance
    model.objective = minimize(xsum(c[i][j] * x[i][j] for i in V for j in V))

    # constraint : leave each city only once
    for i in V:
        model += xsum(x[i][j] for j in V - {i}) == 1

    # constraint : enter each city only once
    for i in V:
        model += xsum(x[j][i] for j in V - {i}) == 1

    # subtour elimination
    for (i, j) in product(V - {0}, V - {0}):
        if i != j:
            model += y[i] - (n + 1) * x[i][j] >= y[j] - n

    # optimizing
    model.optimize()

    # checking if a solution was found
    if model.num_solutions:
        out.write('route with total distance %g found: %s'
                  % (model.objective_value, places[0]))
        nc = 0
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            out.write(' -> %s' % places[nc])
            if nc == 0:
                break
        out.write('\n')
