import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mip import Model, xsum, minimize, BINARY


# Plot Parameters
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['lines.linewidth'] = 1.5


url = 'https://raw.githubusercontent.com/Erik-Chan/Crude-Oil-Data/master/Data/Cleaned_WTI_WSC.csv'

df = pd.read_csv(url)

spread = df['WTI_WCS_diff'] - 5
# 5 is pretty much the spread of heavy and light crude oil at same place
# then the spread now is combination of the transportation cost (pipeline) and the congestion surcharge
#print(min(spread),max(spread))
#plt.plot(spread)
#plt.show()

nodes = ['Hardisty', 'Cushing']
#trans_cost = [6.35]
# transportation cost from Hardisty to Cushing is 5.65-7.05 by Enbridge and 6.35 to 10.46 by Keystone
# choose the midpoint of Enbridge
beta = 0.4   # percentage of the time with congestion
M = 100   # a reasonable upperbound for congestion surcharge
T = len(spread)
b = np.arange(0,1,0.05)

omegasum = []
myb = []

for beta in b:

    model = Model()
    # define the variables
    eps = [model.add_var(lb=-20, ub=20) for t in range(T)]
    omega = [model.add_var(lb=0, ub=50) for t in range(T)]
    alpha = model.add_var(lb=0, ub=20)
    psi = [model.add_var(var_type=BINARY) for t in range(T)]
    trans_cost = model.add_var(lb=4)
    # objective function
    model.objective = minimize(alpha)

    #constraints
    for t in set(range(T)):
        model += spread[t] -trans_cost - eps[t] - omega[t] == 0, 'price decomposition'
        model += eps[t] + alpha >= 0, 'boundary for eps is [-alpha, alpha]'
        model += eps[t] -alpha <=0
        model += eps[t]-alpha + (1-psi[t]) * M >= 0
        model += omega[t]-psi[t]*M <= 0

    model += xsum(psi[t] for t in range(T)) <= int(beta*T)

    # optimizing
    model.optimize()

    if model.num_solutions:
        print('optimal solution cost {} found'.format(model.objective_value))
        print('optimal trans. cost {} found'.format(trans_cost.x))
        omega1 = [omega[i].x for i in range(T)]
        print('congestion surcharge found: %s' % omega1)
        print('The alpha_s is: ', alpha.x)
        print('The omega is: ', [omegaopt.x for omegaopt in omega])
        print('The eps is: ', [epsopt.x for epsopt in eps])
        print('The psi is: ', [psiopt.x for psiopt in psi])
        print('The sum of psi is: ', sum([psiopt.x for psiopt in psi]), 'Compared to beta*T:', beta*T)
        print('The sum of omega is: ', sum(omega1))
        omegasum.append(sum(omega1))
        myb.append(beta)
    #plt.plot(spread)
    #plt.plot(omega1)
    #plt.show()
    #
