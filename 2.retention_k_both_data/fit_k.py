# -*- coding: utf-8 -*-

"""

Fit conductivity


"""


import re
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


h_dry_file = r"./experiment_data_ret_k_fit/1300_k_.txt"

h_dry_file = r"./experiment_data_ret_k_fit/2002_k_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/2020_k_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/4870_k_.txt"


thetaR = 0.05357356257949457
thetaS = 0.5039976509492444
alpha  = 2.3866144032093586
n      = 1.2155865284267393
b      = 0.1623560805267877



thetaR = 2.275928969327872e-09
thetaS = 0.38549938116906224
alpha  = 2.276582585083581
n      = 1.2650886858500179
b      = 0.14384059371552282


thetaR = 3.1717839830150255e-22
thetaS = 0.4252446630671786
alpha  = 0.6202346118730148
n      = 1.2338388635385582
b      = 0.15054283929664944


#b      = 0.0



###############################
########## FUNCTIONS ##########
###############################



def read_dat(file):
    dat = []
    with open(file, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                theta, K = re.split(r"[,\t]", line)
                dat.append([float(theta), float(K)])
    return np.array(dat)


def krel_approx(Se, n, b, lam):
    term = 1.0 - Se**(n / (n * (1.0 + b) - 1.0))
    return Se**lam * (1.0 - term**((n - 1.0) / n))**2


def conductivity(theta, thetaR, thetaS, n, b, lam, ks):
    Se = (theta - thetaR) / (thetaS - thetaR)
    Se = np.clip(Se, 0.0, 1.0)
    return ks * krel_approx(Se, n, b, lam)

#def residuals_lam(par, theta_obs, K_obs, thetaR, thetaS, n, b, ks):
#    lam = par[0]
#    K_pred = conductivity( theta_obs, thetaR, thetaS, n, b, lam, ks)
#    return (K_pred - K_obs) / ks


def residuals_lam(par, theta_obs, K_obs,  thetaR, thetaS, n, b, ks):
    lam = par[0]
    K_pred = conductivity(  theta_obs, thetaR, thetaS, n, b, lam, ks)
    return np.log10(K_pred) - np.log10(K_obs)



# fitting

dat = read_dat(h_dry_file)

theta_obs = dat[:, 0]
K_obs = dat[:, 1]

mask = K_obs > 0.0
theta_fit = theta_obs[mask]
K_fit_obs = K_obs[mask]

ks = np.max(K_obs)


# calc error

fit = least_squares( residuals_lam,  x0=[0.5], bounds=(0.0, 10.0),  args=(  theta_fit,  K_fit_obs,  thetaR,  thetaS,    n,  b,  ks   ))

lam_fit = fit.x[0]


K_pred = conductivity(  theta_obs,  thetaR, thetaS,    n, b, lam_fit, ks)

RMSD = np.sqrt(np.mean((K_pred - K_obs)**2))

print("lambda =", lam_fit)
print("Ks     =", ks, "m/s")
print("RMSD   =", RMSD, "m/s")


theta_plot = np.linspace(np.min(theta_obs), np.max(theta_obs), 500)

K_curve = conductivity(
    theta_plot,
    thetaR, thetaS,
    n, b, lam_fit, ks
)

plt.figure(figsize=(8, 5))

plt.plot(theta_plot, K_curve, label=f'Fitted model (lambda = {lam_fit:.4f})')
plt.scatter(theta_obs, K_obs, label='Observed data')

plt.yscale('log')

plt.xlabel('Water content, θ [m³ m⁻³]')
plt.ylabel('Hydraulic conductivity, K [m s⁻¹]')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
















