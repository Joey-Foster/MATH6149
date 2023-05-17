import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats

# Global parameters
B=1                                    # magnetic field strength
mu = 3e-3                              # blood viscosity 
a=0.5*2e-6                             # particle radius
rho = 5.1e3                            # magnetite density
M_magnetite = 4/3*np.pi*a**3*rho*0.1   # mass of magnetite
M_particle = 3*M_magnetite             # mass of particle
m= 83* M_magnetite                     # magnetic moment

# Blood vessel parameters: [diameter of vessel, length of vessel, bloodstream velocity]
Artery = [3e-3,1e-1,1e-1]
Arteriole = [3e-5,7e-4,1e-2]
Capillary = [7e-6,6e-4,7e-4]
Venule = [4e-5,8e-4,4e-3]
Vein = [5e-3,1e-1,1e-1]

h, L, U_0 = Vein


alpha = M_particle*U_0/(np.pi*mu*a*L)
beta = (m*B/1e-1)/(36*np.pi*mu*a*U_0)
gamma = L/h * beta
print(alpha, beta, gamma) # Cross-check to make sure alpha << beta and gamma

N = 1000 # number of particles 

# Parameteric defintion of trajectories
def x(t,y_0):
    return -gamma**2*t**3/3 + gamma*(1-2*y_0)/2*t**2+(beta+y_0-y_0**2)*t
    
def y(t,y_0):
    return gamma*t+y_0

# Define custom pdf based on velocity profile
class pdf_gen(scipy.stats.rv_continuous):
    def _pdf(self,y):
        return 6*y*(1-y)  # Normalised parabolic pdf
pdf = pdf_gen(a=0,b=1)


# Plot a sample of possible trajectories
fig,ax = plt.subplots(1,2,figsize=(10,5))
plt.figure(1)
y_0s = pdf.rvs(size=N)
t = np.linspace(0,20,5000)
for y_0 in y_0s[::N//20]:
    # maxIndex code to stop trajectories at the upper wall written by Sam
    xVals = x(t,y_0)
    yVals = y(t,y_0)
    try :
        maxIndex = next(i for i, val in enumerate(yVals) if val > 1)
    except: 
        maxIndex = len(yVals)

    colour='g'
    for val in xVals:
        if val>1:
            colour='r'
            break
    
    ax[0].plot(xVals[:maxIndex],yVals[:maxIndex],c=colour)
    ax[0].plot([0,1],[1,1],'--k')
    ax[0].plot([0,1],[0,0],'--k')
ax[0].grid()
ax[0].set_ylim(-0.1,1.1)
ax[0].set_xlim(0,1)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Sample trajectories')

# Plot of all possible trajectories coloured by velocity upon entry
t = np.linspace(0,50,10000)
custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["blue","red","blue"])
all_y_0s = np.linspace(0,1,150)
colours=custom_cmap(np.linspace(0,1,150))
for i,y_0 in enumerate(all_y_0s):
    xVals = x(t,y_0)
    yVals = y(t,y_0)
    try :
        maxIndex = next(x for x, val in enumerate(yVals) if val > 1)
    except: 
        maxIndex = len(yVals)
    ax[1].plot(xVals[:maxIndex],yVals[:maxIndex],color=colours[i])
    ax[1].plot([0,1],[1,1],'--k')
    ax[1].plot([0,1],[0,0],'--k')
ax[1].grid()
ax[1].set_ylim(-0.1,1.1)
ax[1].set_xlim(0,1)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].set_title('All trajectories')
plt.tight_layout()

# Determine how many particles get absorbed into the upper wall
N=10000
trials = 5
average = np.zeros(trials)
for j in range(trials):
    y_0s = pdf.rvs(size=N)
    absorbed = np.zeros_like(y_0s)
    for i, y_0 in enumerate(y_0s):
        for t_i in t:
            if abs(y(t_i,y_0)-1)<1e-2 and x(t_i,y_0)<1:
                absorbed[i] = 1
    success_ratio = sum(absorbed)/N
    print(f'{success_ratio*100}%')
    average[j] = success_ratio
averaged_success_ratio = sum(average)/trials
print(f'{averaged_success_ratio*100}%')
plt.show()