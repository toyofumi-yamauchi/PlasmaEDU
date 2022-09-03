#%%
import ode
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

def fun(t,y):
    ydot = y - t**2 + 1
    return ydot

def main():
    tn   = np.linspace( 0.0, 2.0, 25)     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, tn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, tn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, tn, y0 )         # Runge-Kutta 4
    y_an = tn**2 + 2.0*tn + 1.0 - 0.5*np.exp(tn) # Analytical

    plt.figure(figsize = (6,3.9))
    plt.plot( tn, y_ef, 'ro-', label='Forward Euler (1st)' )
    plt.plot( tn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( tn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( tn, y_an, 'k-',  label='Analytical Solution' )
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Comparison of ODE method: y = t^2 + 2*t + 1 - 0.5*exp(t) \n (run by Toyo at '+current_time+')')
    plt.legend(loc='best')
    plt.xticks(np.arange(0,np.max(tn)+0.25,0.25))
    plt.yticks(np.arange(0,6,0.5))
    plt.grid()
    plt.savefig('ex01_ode_solution.png',dpi=150)
    plt.show()

if __name__ == '__main__':
    main()
