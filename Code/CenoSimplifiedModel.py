import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
import itertools
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY, Var

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)

url = 'https://raw.githubusercontent.com/Erik-Chan/Crude-Oil-Data/master/Data/Cleaned_WTI_WSC.csv'

df = pd.read_csv(url)

df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df.sort_values(by=['DateTime'])

column_names = ['DateTime', 'WTI', 'WCS', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']

df = df.reindex(columns=column_names)

priceData = df[['DateTime', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']]


# This is just a switch to disable the optimization part of the code when testing things to reduce the runtime
MSE = 1

if MSE:
    # time in days
    T = range(len(priceData))

    # The set \mathcal{S} as in the paper
    verts = ['Cushing']
    S = range(len(verts))
    localPrices = priceData[['WCS_Interpolated', 'WTI_Interpolated']].to_numpy()
    localPrices[:, 0] = np.round(localPrices[:, 0], 2)

    # Initialize optimization model object
    model = Model()

    ###################################################################################################################
    # Constants
    ###################################################################################################################
    M = np.max(localPrices) - np.min(localPrices)

    beta = 1

    eta_t = localPrices[:, 0]
    print('My eta_t are:', eta_t)
    plt.plot(eta_t)

    # plt.show()

    lambda_t = localPrices[:, 1]
    print('My lambda_t are:', lambda_t)
    plt.plot(lambda_t)
    # plt.show()
    ###################################################################################################################
    # Variables
    ###################################################################################################################
    #These are the eta_t
    #eta_t = np.array([model.add_var() for t in T])

    # These are the alpha_s
    alpha_s = np.array([model.add_var() for s in S])

    # These are the rho_s
    rho_s = np.array([model.add_var() for s in S])

    # These are the eps_st
    eps_st = np.array([[model.add_var() for t in T] for s in S])

    # These are the w_st
    w_st = np.array([[model.add_var() for t in T] for s in S])

    # These are the psi^t
    psi_t = np.array([model.add_var(var_type=BINARY) for t in T])

    # These are the gamma_s^t
    gamma_st = np.array([[model.add_var(var_type=BINARY) for t in T] for s in S])

    # These are the pi_st
    pi_st = np.array([[model.add_var(var_type=BINARY) for t in T] for s in S])
    ###################################################################################################################
    # Constraints
    ###################################################################################################################

    for s in S:
        for t in T:
            # These are constraints (18b)
            model += eta_t[t] + rho_s[s] + eps_st[s][t] + w_st[s][t] == lambda_t[t]

            # Constraint (18c)
            model += eps_st[s][t] >= -alpha_s[s]

            # Constraint (18d)
            model += eps_st[s][t] <= alpha_s[s]

            # Constraint (18e)
            model += w_st[s][t] <= psi_t[t] * M


    # Constraint (18f)
    model += xsum(psi_t) <= np.floor(beta * len(T))

    for s in S:
        for t in T:
            # Constraint (18g)
            model += w_st[s][t] <= pi_st[s][t] * M

            # Constraint (18h)
            model += eps_st[s][t] + (1 - pi_st[s][t]) * M >= alpha_s[s]

    # Constraint (18i)
    for t in T:
        model += xsum(gamma_st[:, t]) >= psi_t[t]

    # Constraint (18j)
    model += eps_st[s][t] <= -alpha_s[s] + (1 - gamma_st[s][t]) * M

    # Constraint (18k)
    # These are binary constraints see below.

    # Constraint (17l)
    for s in S:
        for t in T:
            model += w_st[s][t] >= 0

    # Set Non-negativity for alpha and rho
    # for s in range(len(verts)):
    # model += alpha_s[s] >= 0
    # model += rho_s[s] >= 2.35

    ###################################################################################################################
    # Optional Constraints and Variables 21a, 21b, 21c
    ###################################################################################################################
    toggle = 0
    if toggle:
        m = 5


        def T_ub(t, m, T):
            return min(len(T), t+m)


        def T_lb(t, m):
            return max(0, t - m)

        # Variable 21c
        nu_t = np.array([model.add_var(var_type=BINARY) for t in T])
        # These end points may need to be say, t_end+1 etc but we get an index out or range
        # if we do that. I think this should be fine.

        # Constraint 21a
        for t in T:
            t_star = t
            t_end = T_ub(t, m, T)
            psi_t_star = [psi_t[_t] for _t in range(t, t_end)]
            model += xsum(psi_t_star) >= nu_t * (len(T) - t_end)

        # Constraint 21b
        for t in T:
            t_star = T_lb(t, m)
            nu_t_star = [nu_t[_t] for _t in range(t_star, t)]
            model += psi_t[t] <= xsum(nu_t_star)

    ###################################################################################################################
    # Objective Function
    ###################################################################################################################

    # Objective function (20a)
    model.objective = minimize(xsum(alpha_s))

    ###################################################################################################################
    # Optimization
    ###################################################################################################################

    # If toggle_optimize != 0, we proceed with the optimization.
    toggle_optimize = 1
    display_parameters = 1
    if toggle_optimize:
        model.optimize()
        if display_parameters:
            print('The solution for alpha_s at the minimum is :', alpha_s[0].x)
            print('The solution for rho_s at the minimum is :', rho_s[0].x)

            for s in S:
                W_list = []
                eps_list = []
                for t in T:
                    W_list.append(w_st[s][t].x)
                    W_list = [round(w, 2) for w in W_list]
                    eps_list.append(eps_st[s][t].x)
                print('The solution for til_W at city {} is:'.format(s), W_list)
                print('The solution for eps_st at node s is:'.format(s), eps_list)
                print('The sum of eps at node s is:'.format(s), sum(eps_list))
            psi_list = []
            for t in T:
                psi_list.append(psi_t[t].x)
            print('The solution for psi^{t} is:', psi_list)
            print('The sum of psi is:', np.sum(psi_list))

            flat_gamma = [item for sublist in gamma_st for item in sublist]
            flat_gamma = [gam.x for gam in flat_gamma]
            print('The sum of the gamma are:', sum(flat_gamma))
