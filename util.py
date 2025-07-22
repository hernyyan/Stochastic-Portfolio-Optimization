import numpy as np
import jax
import jax.numpy as jnp
import simul
import grad

#Assumptions
a = 6
b = 0.12
R = 7
w0 = 10
c = R * (a - b)

def util_grad(x):

    def utility(x):
        w = simul.simulation(x)
        return a*w - R*(a-b) * jnp.log(1 + jnp.exp((w - w0)/R))

    gradient_fn = grad.calculator(utility)
    return gradient_fn(x)