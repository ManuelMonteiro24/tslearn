from math import exp, cos, sqrt
import numpy as np
import matplotlib.pyplot as plt

#grupo 52
#bonzas 78938
#manel 	79138
#carlos 73602
#maga 79032
# S = 0.d1d2d3d4d5d6 * 10^6 = 244468 (soma dos numeros do grupo
# d1=2 d2=4 d3=4 d4=4 d5=6 d6=8
# L = 0.d1d5d6 * 10^1 = 0.268 * 10^1
# R = 0.d1d4d5 * 10^2 = 0.246 * 10^2
# C = 0.d1d2d3 * 10^(-3) = 0.244 * 10^(-3)

G = 52
L = 0.268 * 10**(1)
R = 0.246 * 10**(2)
C = 0.244 * 10**(-3)

def q_function(t):
    return G* exp(-(t*(R/(2*L)))) * cos(t*(sqrt((1/(L*C))-((R**2)/(4*(L**2))))))


# alinea 1(a)
t1 = np.arange(0.0, 1.0, 0.1)
plt.plot(t1, f(t1))
