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
        self.obj_v_sum = 0

    def eval_obj_fun(self, obj_v):
        self.obj_v_sum = np.sum(obj_v)
        return self.obj_v__sum

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
    timeline = np.linspace(0., 1., 100)
    # x = my_ou(x0=1, paths=3, steps=100)(timeline)
    x = my_ou(x0=30, k=2, theta=30, sigma=5, paths=3, steps=100)(timeline)
    # print([row[0] for row in x])
    if plot_toggle:
        plt.plot(x)
        plt.show()

MSE = 1

if MSE:
    # Time
    T = 1
    t = np.linspace(0, T, 1000)
    # The set \mathcal{S}
    verts = Vertex(vertices=['Calgary', 'Hardesty', 'Cushing'])

    edges = Edge(size=3)
    edges.add_edge(1, 2)
    edges.add_edge(1, 3)
    edges.add_edge(2, 3)
    # costs = {e:np.random.uniform(0,1) for e in edges.get_edge()}
    transport_costs = {tuple(e): np.random.uniform(0, 1) for e in edges.get_edge()}
    # transport_costs = {(1,2): }

    # Note that this should eventually look like [[something for _t in t] for s in verts.get_vertex()]
    # This current version just populates placeholder values.

    # This is the \lambda_{s}^{t}
    localPrices = my_ou(x0=30, k=2, theta=30, sigma=5, paths=3, steps=1000)(t)

    model = Model()

    ###################################################################################################################
    # Variables
    ###################################################################################################################

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

    ###################################################################################################################
    # Constraints
    ###################################################################################################################

    # Create the constraints \lambda_{s}^{t} = \eta^{t} + \rho_{s} + \alpha_{s} + \bar{w}_{s}^{t}
    for s in range(len(verts)):
        rho = rho_s[s]
        alpha = alpha_s[s]
        til_W_t = til_W_st[s]
        localPrice_t = [row[s] for row in localPrices]
        for _t in range(len(t)):
            til_W = til_W_t[_t]
            eta = eta_t[_t]
            localPrice = localPrice_t[_t]
            model += eta + rho + alpha + til_W == localPrice

    # Nonnegativity for alpha and rho
    for s in range(len(verts)):
        model += alpha_s[s] >= 0
        model += rho_s[s] >= 0

    # Constraint (20e)
    beta = 0.15
    model += xsum(psi_t) <= np.floor(beta * T)

    # Constraint (20f)
    # Placeholder for M
    M = np.max(localPrices) - np.min(localPrices)

    for s in range(len(verts)):
        til_W_t = til_W_st[s]
        gamma_t = gamma_st[s]
        alpha = alpha_s[s]
        for _t in range(len(t)):
            model += til_W_t[_t] <= -2 * alpha + (1 - gamma_t[_t]) * M

    # Constraint (20g)
    # for _t in range(len(t)):

    # Transposing gamma_st
    gamma_ts = [list(i) for i in zip(*gamma_st)]
    for _t in range(len(t)):
        model += xsum(gamma_ts[_t]) >= psi_t[_t]

    model.objective = minimize(xsum(alpha_s[i] for i in range(len(verts))))
    toggle_optimize = 1
    if toggle_optimize:
        model.optimize()
        print('The solution at the minimum is :', alpha_s[0].x, alpha_s[1].x, alpha_s[2].x)
        print('The solution at the minimum is :', rho_s[0].x, rho_s[1].x, rho_s[2].x)
        eta_list = []
        for _t in range(len(t)):
            eta_list.append(eta_t[_t].x)
        print('The solution at the minimum is:', eta_list)
