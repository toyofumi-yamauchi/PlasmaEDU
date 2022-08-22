################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by a current loop, using its Cartesian components
#
#
################################################################################
#%%
import numpy as np
import bfield
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

# Current Loop
# Ra = 0.05
Ra = 0.01
I0 = 100.
Nturns = 1
Center = np.array([0,0,0])
# Angles = np.array([90,0,0]) * np.pi/180.0
Angles = np.array([30,0,0]) * np.pi/180.0

# X,Y Grid
X = np.linspace(-0.1, 0.1, 50 )
Y = np.linspace(-0.1, 0.1, 50 )

# B-field magnitude
Bnorm = np.zeros((X.size,Y.size))
for i in range(0,X.size):
  for j in range(0,Y.size):
    Point = np.array([ X[i], Y[j], 0.0 ])
    Bx,By,Bz = bfield.loopxyz(Ra,I0,Nturns,Center,Angles,Point)
    Bnorm[i][j] = np.sqrt(Bx*Bx + By*By + Bz*Bz)

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('B-field magnitude [T] of a Current Loop \n (run by Toyo at '+current_time+')')
plt.savefig('ex03_plot_loopxyz_magnitude (run by Toyo at '+current_time+').png',dpi=150)
plt.show()
