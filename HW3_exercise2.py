#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '/Users/toyo/Library/CloudStorage/GoogleDrive-ty20@illinois.edu/My Drive/NPRE598 Computational Plasma Physics/PlasmaEDU/ode/python/')
import ode

def f(t,x,v):
    x_dot = v
    return x_dot
def g(t,x,v):
    v_dot = -x
    return v_dot

Omega_t = np.linspace(0,2*2*np.pi,41)
two_pi = 2.0*np.pi

x_ana = np.cos(Omega_t)
x_ana = np.reshape(x_ana,(len(x_ana),1 ))

xv_rk4 = ode.rk4_2nd(f, g, Omega_t, np.array([1.0,0.0]))
x_rk4 = xv_rk4[:,0]
x_rk4 = np.reshape(x_rk4,(len(x_rk4),1 ))

xv_leap = ode.leapfrog_2nd(f,g,Omega_t,np.array([1.0,0.0]))
x_leap = xv_leap[:,0]
x_leap = np.reshape(x_leap,(len(x_leap),1 ))

Omega_t = np.linspace(0,2*2*np.pi,41)
two_pi = 2.0*np.pi
plt.figure(figsize=(5.5,3.8))
plt.plot(np.linspace(0,2*2*np.pi,101)/two_pi,np.cos(np.linspace(0,2*2*np.pi,101)),'k-',label='Analytical Solution')
plt.plot(Omega_t/two_pi,x_rk4, 'ro',label='Runge-Kutta (4th)')
plt.plot(Omega_t/two_pi,x_leap,'bo',label='Leapfrog (w/o freq. correction)')
#plt.xlim([0,2.0])
plt.xticks(np.arange(0,2.0+0.5,0.5))
plt.xlabel('time, tΩ/2π')
#plt.ylim([-1,1])
plt.yticks(np.arange(-1,1.0+0.5,0.5))
plt.ylabel('position, x')
plt.title('Harmonic Oscillator')
plt.legend(loc='best',framealpha=1)
plt.grid()
plt.tight_layout()
plt.savefig('HW3_exercise2_plot.png',dpi=150)

plt.figure(figsize=(5.5,3.8))
plt.plot(Omega_t/two_pi,ode.error_absolute(x_ana,x_rk4),'ro-',label='Runge-Kutta (4th)')
#plt.xlim([0,2.0])
plt.xticks(np.arange(0,2.0+0.5,0.5))
plt.xlabel('time, tΩ/2π')
plt.ylabel('Absolute error')
plt.title('Absolute Error')
plt.legend(loc='best',framealpha=1)
plt.grid()
plt.tight_layout()
plt.savefig('HW3_exercise2_error.png',dpi=150)