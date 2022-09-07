#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '/Users/toyo/Library/CloudStorage/GoogleDrive-ty20@illinois.edu/My Drive/NPRE598 Computational Plasma Physics/PlasmaEDU/ode/python/')
import ode

Ωt = np.linspace(0,2*2*np.pi,101)
two_pi = 2.0*np.pi

Ω = 1
def harmonic_oscillator(x,Ω):
    x_dot_dot = Ω**2*x
    return x_dot_dot
x_ana = np.cos(Ωt)
x_ana = np.reshape(x_ana,(len(x_ana),1))
x_rk4 = ode.rk4(harmonic_oscillator,Ωt,np.array([0]))

plt.figure(figsize=(5.5,3.8))
plt.plot(Ωt/two_pi,x_ana,'k-',label='Analytical Solution')
plt.plot(Ωt/two_pi,x_rk4,'ro',label='Runge-Kutta (4th)')
plt.xlim([0,2.0])
plt.xticks(np.arange(0,2.0+0.5,0.5))
plt.xlabel('time, tΩ/2π')
plt.ylim([-1,1])
plt.ylabel('position, x')
plt.title('Harmonic Oscillator')
plt.legend(loc='best',framealpha=1)
plt.grid()
plt.tight_layout()
plt.savefig('HW3_exercise2_plot.png',dpi=150)

plt.figure(figsize=(5.5,3.8))
plt.plot(Ωt/two_pi,ode.error_absolute(x_ana,x_rk4),'r-',label='Runge-Kutta (4th)')
plt.xlim([0,2.0])
plt.xticks(np.arange(0,2.0+0.5,0.5))
plt.xlabel('time, tΩ/2π')
plt.ylabel('Absolute error')
plt.title('Absolute Error')
plt.legend(loc='best',framealpha=1)
plt.grid()
plt.tight_layout()
plt.savefig('HW3_exercise2_error.png',dpi=150)