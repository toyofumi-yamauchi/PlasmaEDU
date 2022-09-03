#%%
import ode
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S %p")
print(current_time)

def fun(x,y):
    ydot = y - x**2 + 1
    return ydot

def main():
    xn   = np.linspace( 0.0, 5.0, 20)     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, xn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, xn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, xn, y0 )         # Runge-Kutta 4
    y_an = xn**2 + 2.0*xn + 1.0 - 0.5*np.exp(xn) # Analytical
    y_an = np.reshape(y_an,(len(xn),1))

    plt.figure(figsize = (6,3.9))
    plt.plot( xn, y_an, 'k-',  label='Analytical Solution' )
    plt.plot( xn, y_ef, 'ro', label='Forward Euler (1st)' )
    plt.plot( xn, y_mp, 'go', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, y_rk, 'bo', label='Runge-Kutta (4th)' )
    plt.plot( xn, ode.error_absolute(y_an, y_ef),'r--',label='Absolute error in Forward Euler (1st)')
    plt.plot( xn, ode.error_absolute(y_an, y_mp),'g--',label='Absolute error in Explicit Mid-Point (2nd)')
    plt.plot( xn, ode.error_absolute(y_an, y_rk),'b--',label='Absolute error in Runge-Kutta (4th)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of ODE method: y = x^2 + 2*t + 1 - 0.5*exp(t) \n (run by Toyo at '+current_time+')')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('IVD_exercise_problem.png',dpi=150)
    plt.show()

if __name__ == '__main__':
    main()