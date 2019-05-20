import numpy as np
import matplotlib.pyplot as plt
import math
"""
fig, ax = plt.subplots()
t_data = []
x_data = []
ln, = plt.plot(t_data,x_data, 'ro', animated = True)
"""

def string_profile(t,k_max):
    """String displacement from x = 0 to x = 1"""
    u = np.zeros(1000)
    x = np.arange(0,1,0.001)
    for k in range(0,k_max):
        b_k = (8*((-1)**k))/((math.pi*(2*k+1))**2)
        u = u + b_k * np.sin((2*k+1)*math.pi*x) * np.cos((2*k+1)*math.pi*t)
    return (x,u)

if __name__ == "__main__":

    x,u = string_profile(0.25,10)
    plt.plot(x,u)
    plt.show()