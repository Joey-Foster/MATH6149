import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Please excuse how inefficently designed the code is,
# there is 100% a better way to do the logic, but this works good enough


#define varaiables
A,g,f,alpha,theta,a,l0 = sp.symbols('A g f alpha theta a l_0')

#define l(A)
l_rect = l0 + 2*A/l0
l_tri = 2*sp.sqrt(2*A/sp.sin(theta))
x0 = (3*A/(4*a))**sp.Rational(1,3)
l_para = x0*sp.sqrt(1+4*a**2*x0**2)+1/(2*a)*sp.asinh(2*a*x0)

#compute Q, Q' and Q'' in each case
Q_rect = sp.sqrt(g*sp.sin(alpha)/f)*A**sp.Rational(3,2)/sp.sqrt(l_rect)
Q_rect_prime = sp.diff(Q_rect,A)
Q_rect_double_prime = sp.diff(Q_rect_prime,A)

Q_tri = sp.sqrt(g*sp.sin(alpha)/f)*A**sp.Rational(3,2)/sp.sqrt(l_tri)
Q_tri_prime = sp.diff(Q_tri,A)
Q_tri_double_prime = sp.diff(Q_tri_prime,A)

Q_para = sp.sqrt(g*sp.sin(alpha)/f)*A**sp.Rational(3,2)/sp.sqrt(l_para)
Q_para_prime = sp.diff(Q_para,A)
Q_para_double_prime = sp.diff(Q_para_prime,A)

#lambdify
Q_rect_func = sp.lambdify([A,g,f,alpha,l0],Q_rect,'numpy')
Q_rect_prime_func = sp.lambdify([A,g,f,alpha,l0],Q_rect_prime,'numpy')
Q_rect_double_prime_func = sp.lambdify([A,g,f,alpha,l0],Q_rect_double_prime,'numpy')

Q_tri_func = sp.lambdify([A,g,f,alpha,theta],Q_tri,'numpy')
Q_tri_prime_func = sp.lambdify([A,g,f,alpha,theta],Q_tri_prime,'numpy')
Q_tri_double_prime_func = sp.lambdify([A,g,f,alpha,theta],Q_tri_double_prime,'numpy')

Q_para_func = sp.lambdify([A,g,f,alpha,a],Q_para,'numpy')
Q_para_prime_func = sp.lambdify([A,g,f,alpha,a],Q_para_prime,'numpy')
Q_para_double_prime_func = sp.lambdify([A,g,f,alpha,a],Q_para_double_prime,'numpy')

#evaluate at all parameters except A
def Q_rect_func_one_arg(A):
    return Q_rect_func(A,g,f,alpha,l0)

def Q_tri_func_one_arg(A):
    return Q_tri_func(A,g,f,alpha,theta)

def Q_para_func_one_arg(A):
    return Q_para_func(A,g,f,alpha,a)

def Q_rect_prime_func_one_arg(A):
    return Q_rect_prime_func(A,g,f,alpha,l0)

def Q_tri_prime_func_one_arg(A):
    return Q_tri_prime_func(A,g,f,alpha,theta)

def Q_para_prime_func_one_arg(A):
    return Q_para_prime_func(A,g,f,alpha,a)

def Q_rect_double_prime_func_one_arg(A):
    return Q_rect_double_prime_func(A,g,f,alpha,l0)

def Q_tri_double_prime_func_one_arg(A):
    return Q_tri_double_prime_func(A,g,f,alpha,theta)

def Q_para_double_prime_func_one_arg(A):
    return Q_para_double_prime_func(A,g,f,alpha,a)



def initial_data(x):
    return 2*np.exp(-0.1*x**2)

def initial_data_prime(x):
    return 2*np.exp(-0.1*x**2)*(-0.2*x)

#parameters
g = 9.81
f = 0.1
alpha = 10*np.pi/180
theta = 90*np.pi/180
a = 1/20
l0 = 2

#plot charactersitcs
x0s = np.linspace(-2,10,15)
x = np.linspace(-2,15,500)
fig, ax = plt.subplots(1,3,figsize=(10,5))
for x0 in x0s:
    t_rect = 1/Q_rect_prime_func_one_arg(initial_data(x0))*(x-x0)
    t_tri = 1/Q_tri_prime_func_one_arg(initial_data(x0))*(x-x0)
    t_para = 1/Q_para_prime_func_one_arg(initial_data(x0))*(x-x0)
    ax[0].plot(x,t_rect)
    ax[1].plot(x,t_tri)
    ax[2].plot(x,t_para)

ax[0].set_title(rf'Rectangular, $l_0 = {np.around(l0,2)}$m')
ax[0].grid()
ax[0].set_ylim(0,4)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$t$')

ax[1].set_title(rf'Triangular, $\theta = {np.around(theta*180/np.pi,2)}^o$')
ax[1].grid()
ax[1].set_ylim(0,4)
ax[1].set_xlabel(r'$x$')

ax[2].set_title(rf'Parabolic, $a ={np.around(a,2)}$')
ax[2].grid()
ax[2].set_ylim(0,4)
ax[2].set_xlabel(r'$x$')

plt.suptitle('Characteristics')
plt.tight_layout()
plt.savefig("characteristics.pdf", format="pdf", bbox_inches="tight")

def rect_shock_stats():
    denominator = -Q_rect_double_prime_func_one_arg(initial_data(x0s))*initial_data_prime(x0s)
    time = 1/max(denominator)
    first_associated_x0 = x0s[np.argmax(denominator)]
    AL = initial_data(first_associated_x0-0.00001)
    AR = initial_data(first_associated_x0+0.00001)
    speed = (Q_rect_func_one_arg(AL)-Q_rect_func_one_arg(AR))/(AL-AR)
    return [time,speed]

def tri_shock_stats():
    denominator = -Q_tri_double_prime_func_one_arg(initial_data(x0s))*initial_data_prime(x0s)
    time = 1/max(denominator)
    first_associated_x0 = x0s[np.argmax(denominator)]
    AL = initial_data(first_associated_x0-0.00001)
    AR = initial_data(first_associated_x0+0.00001)
    speed = (Q_tri_func_one_arg(AL)-Q_tri_func_one_arg(AR))/(AL-AR)
    return [time,speed]

def para_shock_stats():
    denominator = -Q_para_double_prime_func_one_arg(initial_data(x0s))*initial_data_prime(x0s)
    time = 1/max(denominator)
    first_associated_x0 = x0s[np.argmax(denominator)]
    AL = initial_data(first_associated_x0-0.00001)
    AR = initial_data(first_associated_x0+0.00001)
    speed = (Q_para_func_one_arg(AL)-Q_para_func_one_arg(AR))/(AL-AR)
    return [time,speed]

print(f"[Shock initiation time, shock speed] for (rect,tri,para): {(rect_shock_stats(),tri_shock_stats(),para_shock_stats())}")

plt.show()