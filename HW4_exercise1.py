#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

freq_correction = False

def frequency_correction(x):
    if freq_correction == True:
        if x == 0:
            alpha = 1
        else:
            alpha = np.tan(x)/x
    else:
        alpha = 1
    return alpha

def boris_bunemann(t,X0,dt,q,m):
    # Birdsall notation
    qmdt2 = q/m*dt/2
    #print('q*Δt/2m = {:.2f}'.format(qmdt2))
    N = np.size(time)
    M = np.size(X0)
    X = np.zeros((N,M))
    X[0,:] = X0
    x = X[0,0]
    y = X[0,1]
    z = X[0,2]
    v_x = X[0,3]
    v_y = X[0,4]
    v_z = X[0,5]
    for n in range(0,N-1):
        # Local field
        E_x,E_y,E_z = Efield(x,y,z)
        B_x,B_y,B_z = Bfield(x,y,z)
        #print(np.tan(qmdt2*B_x))
        #print(np.tan(qmdt2*B_y))
        #print(np.tan(qmdt2*B_z))
        
        # frequency correction factor
        alpha_x = frequency_correction(qmdt2*B_x)
        alpha_y = frequency_correction(qmdt2*B_y)
        alpha_z = frequency_correction(qmdt2*B_z)
        #print('α_x = {:.2f}'.format(alpha_x))
        #print('α_y = {:.2f}'.format(alpha_y))
        #print('α_z = {:.2f}'.format(alpha_z))
        
        # Step 1: Half acceleration (E-field)
        v_minus_x = v_x + qmdt2*E_x*alpha_x
        v_minus_y = v_y + qmdt2*E_y*alpha_y
        v_minus_z = v_z + qmdt2*E_z*alpha_z
        v_minus_vector = np.array((v_minus_x,v_minus_y,v_minus_z))
        # Step 2: B-field rotation
        t_x = qmdt2*B_x*alpha_x
        t_y = qmdt2*B_y*alpha_y
        t_z = qmdt2*B_z*alpha_z
        t_mag = np.sqrt(t_x**2 + t_y**2 + t_z**2)
        t_vector = np.array((t_x,t_y,t_z))
        
        s_x = 2.0*t_x/(1.0+t_mag**2)
        s_y = 2.0*t_y/(1.0+t_mag**2)
        s_z = 2.0*t_z/(1.0+t_mag**2)
        s_vector = np.array((s_x,s_y,s_z))
        
        v_prime_vector = v_minus_vector + np.cross(v_minus_vector,t_vector)
        v_plus_vector = v_minus_vector + np.cross(v_prime_vector,s_vector)
        
        v_plus_x = v_plus_vector[0]
        v_plus_y = v_plus_vector[1]
        v_plus_z = v_plus_vector[2]
        # Step 3: Half acceleratio (E-field)
        v_x = v_plus_x + qmdt2*E_x*alpha_x
        v_y = v_plus_y + qmdt2*E_y*alpha_y
        v_z = v_plus_z + qmdt2*E_z*alpha_z

        # Step 4: Push position
        x = x + v_x*dt
        y = y + v_y*dt
        z = z + v_z*dt
        
        # Storing the coordinates into X
        X[n+1,0] = x
        X[n+1,1] = y
        X[n+1,2] = z
        X[n+1,3] = v_x
        X[n+1,4] = v_y
        X[n+1,5] = v_z
    return X

def velocity_pushback():
    return

# Field parameters
def Bfield(x,y,z):
    Bx = 0.0
    By = 0.0
    Bz = 0.1 # Tesla
    return Bx,By,Bz

def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0 # V/m
    return Ex,Ey,Ez

# Particle parameters
qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

# Cycltron parameters
B_x, B_y, B_z = Bfield(0.0,0.0,0.0)
B_mag = np.sqrt(B_x**2 + B_y**2 + B_z**2)
omega_c = np.abs(qe)*B_mag/me # cyclotron frequency [rad/s]
tau_c   = 2.0*np.pi/omega_c # cyclotron period [s]
#print('τ_c = {:.2e} s'.format(tau_c))

# Initial Condition
v_x_0 = 0.0
v_y_0 = 1e6
v_z_0 = 0.0
v_mag_0 = np.sqrt(v_x_0**2 + v_y_0**2 + v_z_0**2)
r_L = v_mag_0/omega_c # Larmor radius [m]
x_0 = r_L
y_0 = 0.0
z_0 = 0.0
X0 = np.array((x_0,y_0,z_0,v_x_0,v_y_0,v_z_0))

# Time grid
num_cyclotron = 2
num_steps_per_cyclotron = 15
time = np.linspace(0.0,tau_c*num_cyclotron,num_cyclotron*num_steps_per_cyclotron)
dt = time[1] - time[0]

# Velocity push back
X = boris_bunemann(np.linspace(0.0,tau_c,1), X0, -0.5*dt, -qe, me)
X0 = np.array((x_0,y_0,z_0,X[1,3],X[1,4],X[1,5]))

# Boris Bunemann integration
X = boris_bunemann(time, X0, dt, -qe, me)

# plot
X_ana  = np.cos(np.linspace(0.0,2*np.pi,101))
Y_ana  = np.sin(np.linspace(0.0,2*np.pi,101))
X_true = np.cos(time*2*np.pi/tau_c)
Y_true = np.sin(time*2*np.pi/tau_c)
plt.figure(figsize=(5.5,3.8))
plt.plot(X_ana,Y_ana,'k-',label='Analytical')
#plt.plot(X_true,Y_true,'ko-',label='Analytical')
if freq_correction == True:
    plt.plot(X[:,0]/r_L,X[:,1]/r_L,'b.-',label='Boris-Bunemann w/ frequency correction')
else:
    plt.plot(X[:,0]/r_L,X[:,1]/r_L,'b.-',label='Boris-Bunemann w/o frequency correction')
plt.axis('equal')
plt.xlim([-3.0,3.0])
plt.xticks(np.arange(-3.0,3.0+0.5,0.5))
plt.xlabel('x/$r_L$')
plt.ylim([-2.0,2.5])
plt.yticks(np.arange(-2.0,2.5+0.5,0.5))
plt.ylabel('y/$r_L$')
plt.title('Larmor Gyration\n # of cyclotron = {}, # of steps per cycltron = {}'.format(num_cyclotron,num_steps_per_cyclotron))
plt.legend(loc='upper left',framealpha=1)
plt.grid()
plt.tight_layout()
if freq_correction == True:
    if num_cyclotron > 50:
        plt.savefig('HW4_exercise1_trajectory_w_frequency_correction (large periods).png',dpi=150)
    else:
        plt.savefig('HW4_exercise1_trajectory_w_frequency_correction.png',dpi=150)
else:
    plt.savefig('HW4_exercise1_trajectory_wo_frequency_correction.png',dpi=150)


plt.figure(figsize=(5.5,3.8))
plt.plot(time/tau_c,np.abs(X_true*r_L - X[:,0])/r_L,'g.-',label='Error in x')
plt.plot(time/tau_c,np.abs(Y_true*r_L - X[:,1])/r_L,'y.-',label='Error in y')
#plt.axis('equal')
if num_cyclotron > 50:
    plt.xlim([0.0,num_cyclotron])
    plt.xticks(np.arange(0.0,num_cyclotron+10,10))
else:
    plt.xlim([0.0,num_cyclotron])
    plt.xticks(np.arange(0.0,num_cyclotron+0.5,0.5))
plt.xlabel('t$\omega_c$/2$\pi$')
if freq_correction == True:
    plt.ylim([0,0.020])
    plt.yticks(np.arange(0.0,0.020+0.005,0.005))
else:
    plt.ylim([0,0.20])
    plt.yticks(np.arange(0.0,0.20+0.05,0.05))
plt.ylabel('absolute error [$r_L$]')
if freq_correction == True:
    plt.title('Absolute Error w/ frequency correction\n # of cyclotron = {}, # of steps per cycltron = {}'.format(num_cyclotron,num_steps_per_cyclotron))
else:
    plt.title('Absolute Error w/o frequency correction\n # of cyclotron = {}, # of steps per cycltron = {}'.format(num_cyclotron,num_steps_per_cyclotron))
plt.legend(loc='upper left',framealpha=1)
plt.grid()
plt.tight_layout()
if freq_correction == True:
    if num_cyclotron > 50:
        plt.savefig('HW4_exercise1_error_w_frequency_correction (large periods).png',dpi=150)
    else:
        plt.savefig('HW4_exercise1_error_w_frequency_correction.png',dpi=150)
else:
    plt.savefig('HW4_exercise1_error_wo_frequency_correction.png',dpi=150)