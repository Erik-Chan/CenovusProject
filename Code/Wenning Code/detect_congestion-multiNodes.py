"""using the coefficients form the calibration of the spread of WTI/WCS
simulate 3 price spreads sequence with made-up correlation matrix COR.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY


# Plot Parameters
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['lines.linewidth'] = 1.5


dataframe = pd.read_csv('generateOUpath.csv')
spread1 = dataframe['path1']
spread2 = dataframe['path2']
spread3 = dataframe['path3']
spread = [spread1, spread2, spread3]
# 5 is pretty much the spread of heavy and light crude oil at same place
# then the spread now is combination of the transportation cost (pipeline) and the congestion surcharge
fig, ax = plt.subplots()

plt.plot(spread1, label='refinery1')
plt.plot(spread2, label='refinery2')
plt.plot(spread3, label='refinery3')
plt.title('simulated daily spread for 3 locations')
plt.legend()
plt.show()


nodes = ['refinery1', 'refinery2','refinery3']
#trans_cost = [6.35]
# transportation cost from Hardisty to Cushing is 5.65-7.05 by Enbridge and 6.35 to 10.46 by Keystone
# choose the midpoint of Enbridge
beta = 0.4   # percentage of the time with congestion
M = 100   # a reasonable upperbound for congestion surcharge
T = len(spread1)
print(T)

model = Model()


model.cut_passes = 30
model.infeas_tol = 0.01
model.integer_tol = 0.001

print(model.cut_passes)
print(model.cut_passes)
# define the variables for 3 locations
eps = [[model.add_var(lb=-20, ub=20) for t in range(T)] for i in range(3)]
omega = [[model.add_var(lb=0, ub=50) for t in range(T)] for i in range(3)]
alpha = [model.add_var(lb=0, ub=20) for i in range(3)]
psi = [model.add_var(var_type=BINARY) for t in range(T)]
gamma = [[model.add_var(var_type=BINARY) for t in range(T)] for i in range(3)]
trans_cost = [model.add_var(lb=4) for i in range(3)]
# objective function
model.objective = minimize(xsum(alpha[i] for i in range(3)))

#constrains
for i in range(3):
    for t in set(range(T)):
        model += spread[i][t] - trans_cost[i] - eps[i][t] - omega[i][t] == 0, 'price decompose'
        model += eps[i][t] + alpha[i] >= 0, 'boundary for eps is [-alpha, alpha]'
        model += eps[i][t] - alpha[i] <= 0
        model += eps[i][t] - alpha[i] + (1 - gamma[i][t]) * M >= 0, ''
        model += omega[i][t] - psi[t] * M <= 0

for t in set(range(T)):
    model += xsum(gamma[i][t] for i in range(3)) >= psi[t]

model += xsum(psi[t] for t in range(T)) <= int(beta*T)

# optimizing
model.optimize()


print('optimal solution cost {} found'.format(model.objective_value))
#print('optimal trans. cost {} found'.format(trans_cost.x))
omega1 = [[omega[i][t].x for t in range(T)] for i in range(3)]
print('congestion surcharge found: %s' % omega1)

fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(spread[0])
ax0.set_ylabel('spread 1')
ax1.plot(omega1[0])
ax1.set_ylabel('Congestion surcharge-refinery 1')
plt.show()

fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(spread[1])
ax0.set_ylabel('spread 2')
ax1.plot(omega1[1])
ax1.set_ylabel('Congestion surcharge-refinery 2')
plt.show()

fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(spread[2])
ax0.set_ylabel('spread 3')
ax1.plot(omega1[2])
ax1.set_ylabel('Congestion surcharge-refinery 3')
plt.show()

    
