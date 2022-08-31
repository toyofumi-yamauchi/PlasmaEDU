################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by a current loop, solving Biot-Savart
#
#
################################################################################
#%%
import numpy as np
import matplotlib.pyplot as plt
import bfield
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

# Simple Current Loop, discretized in Npoints
# Ra       = 0.05
# Center   = np.array([0,0,0])
# Angles   = np.array([0,0,0]) * np.pi/180.0
# Npoints  = 500
# filament = bfield.makeloop( Ra, Center, Angles, Npoints )
# filament           [X,     Y,   Z]
# filament = np.array([[-0.05, 0.0, -0.030],  # point 1
#                      [ 0.05, 0.0, -0.030],  # point 2
#                      [ 0.05, 0.0, -0.015],  # point 3
#                      [-0.05, 0.0, -0.015],  # point 4
#                      [-0.05, 0.0,  0.000],  # point 5
#                      [ 0.05, 0.0,  0.000],  # point 6
#                      [ 0.05, 0.0,  0.015],  # point 7
#                      [-0.05, 0.0,  0.015],  # point 8
#                      [-0.05, 0.0,  0.030],  # point 9
#                      [ 0.05, 0.0,  0.030]]) # point 10
Npoints = 21
filament1 = np.array([np.linspace(-0.05, 0.05,Npoints),np.repeat(0.0,Npoints),np.repeat(-0.030,Npoints)])
filament2 = np.array([np.linspace( 0.05,-0.05,Npoints),np.repeat(0.0,Npoints),np.repeat(-0.015,Npoints)])
filament3 = np.array([np.linspace(-0.05, 0.05,Npoints),np.repeat(0.0,Npoints),np.repeat( 0.000,Npoints)])
filament4 = np.array([np.linspace( 0.05,-0.05,Npoints),np.repeat(0.0,Npoints),np.repeat( 0.015,Npoints)])
filament5 = np.array([np.linspace(-0.05, 0.05,Npoints),np.repeat(0.0,Npoints),np.repeat( 0.030,Npoints)])
filament = np.concatenate((filament1,filament2,filament3,filament4,filament5),axis=1)                      
current = np.zeros((len(filament.transpose())-1,1))
current[        0:  Npoints-1] =  2.0
current[  Npoints:2*Npoints-1] = -2.0
current[2*Npoints:3*Npoints-1] =  2.0
current[3*Npoints:4*Npoints-1] = -2.0
current[4*Npoints:5*Npoints-1] =  2.0
#print(current)
X = 0.0
Y = np.linspace( -0.06, 0.06, 101 )
Z = np.linspace( -0.06, 0.06, 101 )
Bnorm = np.zeros((Y.size,Z.size))
point = np.zeros((3,1))
for j in range(0,Y.size):
  for k in range(0,Z.size):
    point[0] = X
    point[1] = Y[j]
    point[2] = Z[k]
    Bx, By, Bz = bfield.biotsavart_discretized_current( filament, current, point )
    Bnorm[j][k] = np.sqrt(Bx*Bx + By*By + Bz*Bz)

plt.figure(1)
YY,ZZ = np.meshgrid(Y,Z)
plt.contourf(np.transpose(YY),np.transpose(ZZ),Bnorm,30)
plt.colorbar()
plt.xlabel('Y [m]')
plt.ylabel('Z [m]')
plt.title('B-field magnitude [T] of a Current Loop \n (run by Toyo at '+current_time+')')
plt.savefig('ex09_plot_filament_simple_loop (run by Toyo at '+current_time+').png',dpi=150)
plt.show()
