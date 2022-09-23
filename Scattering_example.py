#%% 
import math
import numpy as np
import matplotlib as plt
import scipy as sp

#%%
# Simple example of implementation of the Newton-Raphson method, f(x) = e^x - 2
print('===Checking the Newton Raphson method===')
def fun(x):
    return math.exp(x)-2.0
def dfun(x):
    return math.exp(x)

print('f(x) = exp(x)-2')
x0  = 1.0
tol = 0.001
xn  = x0
xn1 = x0 + 10.0*tol
error = abs(xn1-xn)
while (error>tol):
    f  =  fun(xn)
    df = dfun(xn)
    print("f({:.3f}) = {:.3f}".format(xn,f))
    xn1 = xn - f/df
    error = abs(xn1-xn)
    xn = xn1

print("root of f = {:.4f}".format(xn1))
f  =  fun(xn1)
print("f({:.4f}) = {:.4e}".format(xn1,f))
#%% 
# Find the Distance of Closest Approach in a Head-on Collision between 2 particles interacting with potential φ(r)
def distance_of_closest_approach_function(r,p,Er,V):
    return r**2*(1.0-V(r)/Er-p**2)
