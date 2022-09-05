#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

sys.path.insert(1, '/Users/toyo/Library/CloudStorage/GoogleDrive-ty20@illinois.edu/My Drive/NPRE598 Computational Plasma Physics/PlasmaEDU/ode/python/')

import ode

qe  = 1.60217662e-19
me  = 9.10938356e-31
mp  = 1.6726219e-27
mu0 = 4.0*np.pi*1.0e-7
qm  = qe/mp

B0   = 0.5000 # [T]
E0   = 10.0e3 # [V/m]
freq = 1e4    # [Hz]

def Efield(t,x,y,z):
    Ex = 1e4 * np.sin(2.0*np.pi*freq * t)
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez

def Bfield(x,y,z):
    Bx = 0.0
    By = 0.0
    Bz = B0
    return Bx, By, Bz

# RHS of the ODE problem, dy/dx = f(x,y)
def fun(t,X):
    x, y, z, vx, vy, vz = X
    # E-field [V/m]
    Ex, Ey, Ez = Efield(t,x,y,z)
    # B-field [T]
    Bx, By, Bz = Bfield(x,y,z)
    # Newton-Lorentz equation in Cartesian coordinates
    Xdot = np.zeros(6)
    Xdot[0] = vx
    Xdot[1] = vy
    Xdot[2] = vz
    Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
    Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
    Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
    return Xdot


def main():

   # Initial velocity [m/s]
   vy0 = 1.0e4

   # Larmor pulsation [rad/s]
   w_L = np.abs(qm) * B0

   # Larmor period [s]
   tau_L = 2.0*np.pi / w_L

   # Larmor radius [m]
   r_L = vy0 / w_L

   # Initial conditions
   X0 = np.array(( -r_L, 0.0, 0.0, 0.0, vy0, 0.0  ))

   # Number of Larmor gyrations
   N_gyros = 20

   # Time grid
   time = np.linspace( 0.0, tau_L*N_gyros, 25*N_gyros )

   # Runge-Kutta 4
   X = ode.rk4( fun, time, X0 )

   # Collect components
   x  = X[:,0]
   y  = X[:,1]
   z  = X[:,2]
   vx = X[:,3]
   vy = X[:,4]
   vz = X[:,5]

   # Perpendicular velocity
   vp = np.sqrt( vx*vx + vy*vy )

   # Plot 1 - Trajectory
   plt.figure(figsize=(5.5,3.8))
   plt.plot( x[0], y[0], 'o', label='Starting point' )
   plt.plot( x[1], y[1], 'o', label='2nd point' )
   plt.plot( x, y, 'b-', label='RK4' )
   plt.plot( x[-1], y[-1], 'o', label='Last point' )
   plt.xlabel('x [m]')
   plt.ylabel('y [m]')
   plt.title('Polarization drift trajectory (x-y)\n (run by Toyo at '+current_time+')')
   plt.axis('equal')
   plt.legend()
   plt.grid()
   plt.savefig('ex06_polarization_trajectory.png',dpi=150)

    
   plt.figure(figsize=(5.5,3.8))
   plt.plot( time[0]/tau_L, x[0], 'b-', label='Starting point' )
   plt.plot( time[1]/tau_L, x[1], 'b-', label='2nd point' )
   plt.plot( time/tau_L, x, 'b-', label='RK4' )
   plt.plot( time[-1]/tau_L, x[-1], 'b-', label='Last point' )
   plt.xlabel('time / tau_L')
   plt.ylabel('x position [m]')
   plt.title('Polarization drift shift\n (run by Toyo at '+current_time+')')
   plt.grid()
   plt.savefig('ex06_polarization_position.png',dpi=150)


if __name__ == '__main__':
   main()
