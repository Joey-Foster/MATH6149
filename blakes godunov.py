import numpy as np
import sympy as sp
import scipy.integrate
import matplotlib.pyplot as plt

alpha = 1*np.pi/180
sinalpha = np.sin(alpha)
g=9.81
f=0.1
# base of rectangle
l0 = 2
# angle of triangle
theta = 22*np.pi/180
# evaluate sin once since this is constant
sintheta = np.sin(theta)


def lrect(A):
    return l0 + 2 * A  / l0

# not used yet
def ltriangle(A):
    return 2 * np.sqrt(2*A/sintheta)

#parabola param
a = 0.05
def lparabola(A):
    x0 = (3 * A / (4*a))**(1./3)
    return x0 * np.sqrt(1+4*a*a*x0*x0)+np.arcsinh(2*a*x0)/(2*a)

def getQ(lfunc):
    return lambda A: np.sqrt(A**3 * g * sinalpha/(lfunc(A) * f))


# Q(c: float) -> float
# initialC(x: float) -> float
def solveGodunov(maxX, maxT, nIntervals, Q, initialC):
    #xfunc = lambda i: (i * maxX) / nIntervals
    x = np.linspace(0, maxX, nIntervals)
    # constant
    h = x[1] - x[0]
    # t-time, c np array of c_i parts
    def ivp(t, c):
        dcdt = np.empty(len(c))
        for i in range(1, len(c)):
            dcdt[i] = -(Q(c[i]) - Q(c[i-1])) / h
        # inflow
        dcdt[0] = 0
        return dcdt
    return x, scipy.integrate.solve_ivp(ivp, (0, maxT), initialC(x), dense_output=True)

maxT = 15
x, soln = solveGodunov(30, maxT, 1000, getQ(lparabola), lambda x: 2 * np.exp(-0.1*(x-10)**2))

tspace = np.linspace(0, maxT, 20)

# ax1 will be used to find the axes limits for the animation
fig1, ax1 = plt.subplots()
plt.subplots_adjust(top=0.89, right=0.77)
for t in tspace:
    ax1.plot(x,soln.sol(t),label=rf'$t= {np.around(t,2)}$')


plt.xlabel(r'$x$')
plt.ylabel(r'$A$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'$A(x,t)$ in the Lower course')
plt.grid()


# # animation
# import matplotlib.animation as animation
    
# fig2, ax2 = plt.subplots()
# fps = 30
# def plotFrame(frame):
#     ax2.clear()
    
#     t = frame / fps
#     # find axes limits from the static plot
#     ax2.set_ylim(ax1.get_ylim())
    
#     ax2.plot(x,soln_rect.sol(t),label=rf'$t= {np.around(t,2)}$')
#     # label wasn't working and haven't figured out why
#     # displaying the current timestep with the title for now
#     plt.title(rf'$t= {np.around(t,2)}$', loc='left')


# ani = animation.FuncAnimation(fig2, plotFrame, frames=maxT*fps, interval=1000/fps, repeat=True)
plt.show()

# writergif = animation.PillowWriter(fps=fps)
# ani.save('animation.gif',writer=writergif)