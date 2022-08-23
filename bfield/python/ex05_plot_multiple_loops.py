################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by multiple current loops
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

# Loops ( Ra,I0,Nturns, Xcenter,Ycenter,Zcenter, EulerAngles1,2,3 )
EA1 = 45
Loops = np.array([[ 0.05,100,1,  0.04*np.sin(EA1*np.pi/180.0), 0.04*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1,  0.03*np.sin(EA1*np.pi/180.0), 0.03*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1,  0.02*np.sin(EA1*np.pi/180.0), 0.02*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1,  0.01*np.sin(EA1*np.pi/180.0), 0.01*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1,  0.00*np.sin(EA1*np.pi/180.0), 0.00*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1, -0.01*np.sin(EA1*np.pi/180.0),-0.01*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1, -0.02*np.sin(EA1*np.pi/180.0),-0.02*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1, -0.03*np.sin(EA1*np.pi/180.0),-0.03*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ],
                  [ 0.05,100,1, -0.04*np.sin(EA1*np.pi/180.0),-0.04*np.cos(EA1*np.pi/180.0),0, EA1,0,0 ] ])
Nloops = np.size(Loops,0)

mesh_size = 500
X = np.linspace( -0.1, 0.1, mesh_size )
Y = np.linspace( -0.1, 0.1, mesh_size )
Bnorm = np.zeros((X.size,Y.size))

for i in range(0,X.size):
  for j in range(0,Y.size):
    for k in range(0,Nloops):
      Ra     = Loops[k][0]
      I0     = Loops[k][1]
      Nturns = Loops[k][2]
      Center = Loops[k][3:6]
      Angles = Loops[k][6:9] * np.pi/180.0
      Point  = np.array([ X[i], Y[j], 0.0 ])
      Bx,By,Bz = bfield.loopxyz( Ra,I0,Nturns,Center,Angles,Point )
      Bnorm[i][j] += np.sqrt( Bx*Bx + By*By + Bz*Bz )

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
#plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,np.linspace(0,0.032,30))
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('B-field magnitude [T] - Multiple Loops\n EA = {:.0f} deg, mesh size = ({:.0f}x{:.0f})\n (run by Toyo at '.format(EA1,mesh_size,mesh_size)+current_time+')')
plt.savefig('ex05_plot_multiple_loops (run by Toyo at '+current_time+').png',dpi=150)
plt.show()
