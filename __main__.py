import numpy as np
import matplotlib.pyplot as plt
import sgd
import simul
import time


#Logic:

#simulation module takes in x as a vector input, outputs w as a number

#util takes in w, outputs gradient

#gradient module takes in u and x, outputs jacobian

#sgd module takes in an objective to be maximized (utility), the variables to max wrt (x),
#and outputs optimized x and path



#initialize portfolio weights
x = np.array([0.1, 0.1])

x = sgd.sgd_algo(x)

#Outputs
start = time.time()
elapsed = time.time() - start
minutes, seconds = divmod(elapsed, 60)
print(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds")

print(f"Option Price:     {simul.p:.10f}")
print(f"Stock:    {x[0]:.10f}")
print(f"Option:   {x[1]:.10f}")
print(f"Cash:     {(1 - x[0] - x[1]):.10f}")

#Plot
x = list(range(1, sgd.n + 1))
plt.figure(figsize=(10,6))
plt.plot(x, sgd.stock_path, linestyle='-', color='blue', label='stock')
plt.plot(x, sgd.option_path, linestyle='-', color='red', label='option')
plt.xlabel('Iterations')
plt.ylabel('Proportion of Each Asset')
plt.title('Stochastic Optimization Path')
plt.grid(True)
plt.legend()
plt.show()
