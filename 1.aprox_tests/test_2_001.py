# -*- coding: utf-8 -*-

"""

Test conductivity

"""







import numpy as np
from scipy.integrate import quad




# soil data here
#                              theta_r, theta_s,  alpha,       n,             ks 
#loamy sand
thetaR, thetaS, alpha, n, ks = 0.049,    0.39,    3.475361614, 1.745822153,   1.21755E-05
#silty loam
thetaR, thetaS, alpha, n, ks = 0.065,    0.439,   0.505824662, 1.66341265,    2.11099E-06
#silty clay
#thetaR, thetaS, alpha, n, ks = 0.111,    0.481,   1.621810097, 1.321295634,   1.11298E-06


lam = 0.5



theta_min = 1e-6
theta_max = 1.0 - 1e-6
N = 1000

theta = np.linspace(theta_min, theta_max, N)


b = [0.1, 0.5, 1.0, 2.0]

b_values = [0.1, 0.5, 1.0, 2.0]

###############################
########## FUNCTIONS ##########
###############################

## saturation with approximation
def saturation(h, alpha, n, b):
    return (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n + b))

# helper
def integrand(x, alpha, n, b):
    m = 1.0 - 1.0 / n
    C = m * n * alpha**n * x**(n - 1.0) \
        * (1.0 + (alpha * x)**n)**(-m - 1.0)
    return C * np.exp(-b * alpha * x)
# saturation with numerical integration
def saturation_integral(h, alpha, n, b):
    Z = quad( integrand, 0.0, np.inf, args=(alpha, n, b) )[0]
    return quad(  integrand, h, np.inf, args=(alpha, n, b) )[0] / Z

# traditional VGM saturation without b
def saturation_vg(h, alpha, n):
    return (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n))

def krel_approx(theta, n, b, lam):
    term = 1.0 - theta**(n / (n * (1.0 + b) - 1.0))
    return theta**lam * (1.0 - term**((n - 1.0) / n))**2

def inv_h(theta, alpha, n, b):
    q = n / (n * (1.0 + b) - 1.0)
    return alpha * theta**(q / n) \
        * (1.0 - theta**q)**(-1.0 / n)

def krel_integral(theta, alpha, n, b, lam):
    numerator = quad( inv_h, 0.0, theta, args=(alpha, n, b) )[0]
    denominator = quad( inv_h, 0.0, 1.0, args=(alpha, n, b)  )[0]
    return theta**lam * (numerator / denominator)**2


def rmsd(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))



############ CALCULATIONS ###################

for b in b_values:

    k_approx = krel_approx(theta, n, b, lam)

    k_integral = np.array([
        krel_integral(x, alpha, n, b, lam)
        for x in theta
    ])

    error = rmsd(k_approx, k_integral)

    print(f"b = {b:3.1f}, RMSD = {error:.6f}")


























