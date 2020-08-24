import numpy as np
import matplotlib.pyplot as plt
import sdepy

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)


@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k * (x-theta), 'dw': sigma}
T = 1
t = np.linspace(0, T, 655)
x = my_ou(x0= 16.31, k = 0.011928352054776574, theta = 16.31,
          sigma = 1.00597006920309, paths = 1, steps = len(t))(t)

mean = (0, 0, 0, 0)
cov = [[1.0000, 0.8819, 0.8118, 0.5096],
       [0.8819, 1.0000, 0.9744, 0.3065],
       [0.8118, 0.9744, 1.0000, 0.2832],
       [0.5096, 0.3065, 0.2832, 1.0000]]
dW = np.random.multivariate_normal(mean, cov, (655))
print('My DW is: ', dW)
X_0 = [5, 4.5, 5.25, 0]

X_t = [X_0]
k = [0.1, 0.1, 0.1, 0.1]
mu = [5, 4.5, 5.25, 0]

for t in range(5):
    delXt = []
    for i in range(len(X_0)):
        delXt.append(k[i]*(X_t[t][i]-mu[i])+dW[t][i])
    X_t.append(delXt)

X_t = np.array(X_t)
