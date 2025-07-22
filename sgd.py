import numpy as np
import util

rng = np.random.default_rng()

#Assumptions
t = 0.0001 #learning rate
n = 1000000 #number of iterations

stock_path = []
option_path = []

def sgd_algo(x):
    for i in range(n):
        #z = rng.standard_normal()
        grad_step = util.util_grad(x)
        x += t * grad_step
        stock_path.append(x[0])
        option_path.append(x[1])
    return x