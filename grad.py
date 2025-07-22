import jax
import jax.numpy as jnp
import numpy as np

def calculator(fn):

    grad_fn = jax.grad(fn)
    
    def calc(x):
        x_jax = jnp.array(x)  # Convert to jax array
        gradient = grad_fn(x_jax)
        return np.array(gradient)  # Convert back to numpy

    return calc