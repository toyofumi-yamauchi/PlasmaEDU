#%%
import sys
import numpy as np
from pylab import plot, axis, show, savefig, title,legend,xlabel,ylabel
sys.path.insert(1, '/Users/toyo/Library/CloudStorage/GoogleDrive-ty20@illinois.edu/My Drive/NPRE598 Computational Plasma Physics/PlasmaEDU/ode/python/')
import ode
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

# Physical Constants (SI units, 2019 redefinition)
qe   = 1.602176634e-19       # fundamental charge [C]
me   = 9.109383701528e-31    # electron rest mass [kg]
mp   = 1.6726219236951e-27   # proton rest mass [kg]
lux  = 299792458.0           # speed of light [m/s]
hp   = 6.62607015e-34        # Planck constant [Js]
muref= 1.0000000005415e-7    # Reference measure of mu0
mu0  = 4.0*np.pi*muref       # Vacuum permeability [H/m]
eps0 = 1/lux/lux/mu0         # Vacuum permittivity [F/m]
fine = qe*qe*lux*mu0/2.0/hp  # Fine structure
kc   = 1.0/4.0/np.pi/eps0    # Coulomb constant
hbar = hp/2.0/np.pi          # h-bar
epsilon = 1.0e-15            # Small number (nucleus size) [m]

# Bohr model (SI units)
a0   = hbar/me/lux/fine             # Bohr radius
mk   = kc*qe*qe/me
vb   = np.sqrt(mk/a0)               # Bohr speed
tb   = 2.0*np.pi*np.sqrt(a0**3/mk)  # Bohr period

# Number of particles
Np = 2
Np_half = int(Np/2)

# Charge and Mass
q = np.concatenate( (qe*np.ones(Np_half), -qe*np.ones(Np_half) ) )
m = np.concatenate( (mp*np.ones(Np_half),  me*np.ones(Np_half) ) )

# Dynamics
def dynamics(t,Y):

   x = Y[0*Np:1*Np]
   y = Y[1*Np:2*Np]
   z = Y[2*Np:3*Np]

   vx = Y[3*Np:4*Np]
   vy = Y[4*Np:5*Np]
   vz = Y[5*Np:6*Np]

   ax = np.zeros(Np)
   ay = np.zeros(Np)
   az = np.zeros(Np)

   for i in range(Np):
      for j in range(Np):
         if (j!=i):

            x_ij = x[i]-x[j]
            y_ij = y[i]-y[j]
            z_ij = z[i]-z[j]

            r_ij = np.sqrt( x_ij**2 + y_ij**2 + z_ij**2 )

            Fx_ij = kc * q[i] * q[j] * x_ij / (r_ij**3)
            Fy_ij = kc * q[i] * q[j] * y_ij / (r_ij**3)
            Fz_ij = kc * q[i] * q[j] * z_ij / (r_ij**3)

            ax[i] += Fx_ij / m[i]
            ay[i] += Fy_ij / m[i]
            az[i] += Fz_ij / m[i]

   dY = np.concatenate( (vx, vy, vz, ax, ay, az) )
   return dY


def main():

    # Initial State Vector
    Rx = np.array( [0.0, a0] )
    Ry = np.zeros(Np)
    Rz = np.zeros(Np)

    Vx = np.zeros(Np)
    Vy = np.array( [0.0, vb] )
    Vz = np.zeros(Np)

    Y0 = np.concatenate( ( Rx, Ry, Rz, Vx, Vy, Vz ) )

    # Time grid
    tspan = np.linspace(0.0, tb, 40)

    # Solve ODE
    Y = ode.rk4(dynamics, tspan, Y0)

    Rx = Y[ :, 0*Np:1*Np]
    Ry = Y[ :, 1*Np:2*Np]
    Rz = Y[ :, 2*Np:3*Np]

    Vx = Y[ :, 3*Np:4*Np]
    Vy = Y[ :, 4*Np:5*Np]
    Vz = Y[ :, 5*Np:6*Np]

    # Plot results
    plot( Rx[:,0:1], Ry[:,0:1], 'ro-',label='proton')
    plot( Rx[:,1:2], Ry[:,1:2], 'b.-',label='electron')
    axis('equal')
    xlabel('x, m')
    ylabel('y, m')
    legend()
    title('H atom model (1proton-1electron)\n (run by Toyo at {})'.format(current_time))
    savefig('ex10_nbody_Hatom_overall.png',dpi=150)
    show()

    plot( Rx[:,0:1], Ry[:,0:1], 'ro-',label='proton')
    #plot( Rx[:,1:2], Ry[:,1:2], 'b.-',label='electron')
    axis('equal')
    xlabel('x, m')
    ylabel('y, m')
    legend()
    title('H atom model (1proton-1electron)\n (run by Toyo at {})'.format(current_time))
    savefig('ex10_nbody_Hatom_proton.png',dpi=150)
    show()


if __name__ == '__main__':
    main()
