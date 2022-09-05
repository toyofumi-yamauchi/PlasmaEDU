#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

sys.path.insert(1, '/Users/toyo/Library/CloudStorage/GoogleDrive-ty20@illinois.edu/My Drive/NPRE598 Computational Plasma Physics/PlasmaEDU/ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

# Charge-to-mass ratio (q/m)
qm = -qe/mp


def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez


def Bfield(x,y,z):
    B0 = 0.5
    # Major radius (projection)
    R = np.sqrt( x*x + y*y )
    # Sin, Cos of particle position on (x,y) plane
    ca = x/R
    sa = y/R
    # B-field [T]
    Bx = -B0 * sa
    By =  B0 * ca
    Bz =  0.0
    return Bx, By, Bz


def fun(t,X):
   x, y, z, vx, vy, vz = X
   # E-field [V/m]
   Ex, Ey, Ez = Efield(x,y,z)
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
    vy0 = 1.0e6
    vz0 = 1.0e6

    # Magnetic Field at the origin [T]
    Bz = 0.5

    # Larmor pulsation [rad/s]
    w_L = np.abs(qm * Bz)

    # Larmor period [s]
    tau_L = 2.0*np.pi / w_L

    # Larmor radius [m]
    r_L = vy0 / w_L

    # Initial conditions
    X0 = np.array( [ 0.72, 0.0, 0.0, 0.0, vy0, vz0 ] )

    # Number of Larmor gyrations
    N_gyro = 35

    # Number of points per gyration
    N_points_per_gyration = 100

    # Time grid
    time = np.linspace( 0.0, tau_L*N_gyro, N_gyro*N_points_per_gyration )

    # Solve ODE (Runge-Kutta 4)
    X = ode.rk4( fun, time, X0 )

    # Get components of the state vector
    x  = X[:,0]
    y  = X[:,1]
    z  = X[:,2]
    vx = X[:,3]
    vy = X[:,4]
    vz = X[:,5]

    R = np.sqrt(x*x + y*y)

    plt.figure(figsize = (5.5,3.8))
    plt.plot( x[0], y[0], 'o', label='Starting point' )
    plt.plot( x[1], y[1], 'o', label='2nd point' )
    plt.plot( x, y, 'b-', label='Runge-Kutta (4th)' )
    plt.plot( x[-1], y[-1], 'o', label='Last point' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Grad-B drift trajectory (x-y)\n (run by Toyo at '+current_time+')')
    plt.axis('equal')
    plt.legend(loc=3)
    plt.grid()
    plt.savefig('ex03_drift_grad_B_trajectory.png')
    plt.show()

    plt.figure(figsize = (5.5,3.8))
    plt.plot( R[0], z[0], 'o', label='Starting point')
    plt.plot( R[1], z[1], 'o', label='2nd point')
    plt.plot( R, z, 'b-', label='Runge-Kutta (4th)')
    plt.plot( R[-1], z[-1], 'o', label='Last point')
    plt.xlabel('R, Radius [m]')
    plt.ylabel('Z, Vertical Coordinate [m]')
    plt.title('Grad-B drift trajectory (r-z)\n (run by Toyo at '+current_time+')')
    plt.axis('equal')
    plt.grid()
    plt.savefig('ex03_drift_grad_B_vertical_drift.png')
    plt.show()


if __name__ == '__main__':
   main()
