import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm

rng = np.random.default_rng()

#Assumptions
mu = 0.1
sigma = 0.25
T = 1
r = 0.05
S0 = 100
K = S0 * jnp.exp(r*T)

def put_price(S, v, T, K, r):
    d1 = (jnp.log(S/K) + (r + v**2/2)*T) / (v * jnp.sqrt(T))
    d2 = d1 - v * jnp.sqrt(T)
    return K * jnp.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

p = put_price(S0,sigma,T,K,r)

def simulation(x):
    z = rng.standard_normal()
    ST = S0 * jnp.exp((mu - sigma**2 /2)*T + sigma * jnp.sqrt(T)*z)
    pi_stock = ST / S0 #stock return
    pi_option = max(K - ST, 0.0) #option return
    return x[0] * pi_stock + x[1] * pi_option + (1 - x[0] - p*x[1]) * jnp.exp(r*T)