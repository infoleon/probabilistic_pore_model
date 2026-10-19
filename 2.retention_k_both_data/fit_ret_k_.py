# -*- coding: utf-8 -*-

"""

Fit retention and conductivity

"""


import re
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



h_dry_file = r"./experiment_data_ret_k_fit/1300_h_dry_.txt"
h_wet_file = r"./experiment_data_ret_k_fit/1300_h_wet_.txt"
k_file = r"./experiment_data_ret_k_fit/1300_k_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/2002_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/2002_h_wet_.txt"
#k_file = r"./experiment_data_ret_k_fit/2002_k_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/2020_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/2020_h_wet_.txt"
#k_file = r"./experiment_data_ret_k_fit/2020_k_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/4870_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/4870_h_wet_.txt"
#k_file = r"./experiment_data_ret_k_fit/4870_k_.txt"


weight_k = 0.1


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
                theta, h = re.split(r"[,\t]", line)
                dat.append([float(theta), float(h)])
    return np.array(dat)

def read_dat_k(file):
    dat = []
    with open(file, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                theta, K = re.split(r"[,\t]", line)
                dat.append([float(theta), float(K)])
    return np.array(dat)

def retention(h, thetaR, thetaS, alpha, n, b):
    Se = (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n + b))
    return thetaR + (thetaS - thetaR) * Se

def krel_approx(Se, n, b, lam):
    term = 1.0 - Se**(n / (n * (1.0 + b) - 1.0))
    return Se**lam * (1.0 - term**((n - 1.0) / n))**2

def conductivity(theta, thetaR, thetaS, n, b, lam, ks):
    Se = (theta - thetaR) / (thetaS - thetaR)
    Se = np.clip(Se, 1e-12, 1.0)
    return ks * krel_approx(Se, n, b, lam)


def residuals(par, dry_dat, wet_dat, k_dat):
    thetaR, thetaS, alpha, n, b, lam, log10_ks = par
    ks = 10.0**log10_ks

    # retention data
    theta_dry = dry_dat[:, 0]
    h_dry = dry_dat[:, 1]

    theta_wet = wet_dat[:, 0]
    h_wet = wet_dat[:, 1]

    # conductivity data
    theta_k = k_dat[:, 0]
    K_obs = k_dat[:, 1]

    # predicted retention
    theta_dry_pred = retention(   h_dry, thetaR, thetaS, alpha, n, 0.0 )

    theta_wet_pred = retention(h_wet, thetaR, thetaS, alpha, n, b  )

    # predicted conductivity
    K_pred = conductivity( theta_k, thetaR, thetaS, n, b, lam, ks  )

    # residuals
    res_dry = theta_dry_pred - theta_dry
    res_wet = theta_wet_pred - theta_wet

    mask = K_obs > 0.0
    res_K = weight_k * ( np.log10(K_pred[mask]) - np.log10(K_obs[mask]) )
    return np.concatenate(( res_dry,  res_wet, res_K ))




dry_dat = read_dat(h_dry_file)
wet_dat = read_dat(h_wet_file)
dat = read_dat_k(k_file)

# Bounds
theta_max_data = max( np.max(dry_dat[:, 0]), np.max(wet_dat[:, 0]), np.max(dat[:, 0]) )

lower = [
    0.0,        # thetaR
    theta_max_data,        # thetaS
    1e-8,       # alpha
    1.000001,   # n
    0.0,        # b
    -6.0,        # lambda
    -15.0       # log10(Ks)
]

upper = [
    0.6,        # thetaR
    1.0,        # thetaS
    np.inf,     # alpha
    np.inf,     # n
    np.inf,     # b
    10.0,       # lambda
    0.0         # log10(Ks)
]


## Initial guess

thetaR0 = 0.05
thetaS0 = theta_max_data
alpha0 = 1.0
n0 = 1.5
b0 = 0.1
lam0 = 0.5

K_positive = dat[dat[:, 1] > 0.0, 1]
ks0 = np.max(K_positive)

x0 = [ thetaR0, thetaS0, alpha0, n0, b0, lam0, np.log10(ks0)]


fit = least_squares( residuals, x0, bounds=(lower, upper), args=(dry_dat, wet_dat, dat) )

thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit, lam_fit, log10_ks_fit = fit.x

ks_fit = 10.0**log10_ks_fit



# retention plot

# retention curves
h_plot = np.logspace(np.log10(1e-3), np.log10(1.7e3), 1000)

theta_dry_curve = retention(
    h_plot,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, 0.0
)

theta_wet_curve = retention(
    h_plot,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit
)

# retention plot
plt.figure(figsize=(8, 5))

plt.plot(h_plot, theta_dry_curve, label='Drying fit (b = 0)')
plt.plot(h_plot, theta_wet_curve, label=f'Wetting fit (b = {b_fit:.4f})')

plt.scatter(dry_dat[:, 1], dry_dat[:, 0], label='Drying data')
plt.scatter(wet_dat[:, 1], wet_dat[:, 0], label='Wetting data')

plt.xscale('log')

plt.xlabel('Pressure head, h [m]')
plt.ylabel('Water content, θ [m³ m⁻³]')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## conductivity plot

# conductivity curve
theta_plot = np.linspace(np.min(dat[:, 0]), np.max(dat[:, 0]), 500)

K_curve = conductivity( theta_plot, thetaR_fit, thetaS_fit, n_fit, b_fit, lam_fit, ks_fit )

# conductivity plot
plt.figure(figsize=(8, 5))

plt.plot(theta_plot, K_curve, label=f'Conductivity fit (λ = {lam_fit:.4f})')

# only positive K can be shown on log scale
mask = dat[:, 1] > 0.0
plt.scatter(dat[mask, 0], dat[mask, 1], label='Conductivity data')

plt.yscale('log')

plt.xlabel('Water content, θ [m³ m⁻³]')
plt.ylabel('Hydraulic conductivity, K [m s⁻¹]')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




## ERROR handling

# retention predictions
theta_dry_obs = dry_dat[:, 0]
h_dry = dry_dat[:, 1]

theta_wet_obs = wet_dat[:, 0]
h_wet = wet_dat[:, 1]

theta_dry_pred = retention(    h_dry,  thetaR_fit, thetaS_fit, alpha_fit, n_fit, 0.0)

theta_wet_pred = retention(  h_wet,  thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit)

# retention RMSD
rmsd_dry = np.sqrt(np.mean((theta_dry_pred - theta_dry_obs)**2))
rmsd_wet = np.sqrt(np.mean((theta_wet_pred - theta_wet_obs)**2))

rmsd_ret = np.sqrt( np.mean( np.concatenate(( theta_dry_pred - theta_dry_obs, theta_wet_pred - theta_wet_obs ))**2 ) )


# conductivity predictions
theta_k = dat[:, 0]
K_obs = dat[:, 1]

K_pred = conductivity(  theta_k, thetaR_fit, thetaS_fit, n_fit, b_fit, lam_fit, ks_fit )

mask = K_obs > 0.0

# conductivity error in log10 space
rmsd_k_log = np.sqrt( np.mean( ( np.log10(K_pred[mask]) - np.log10(K_obs[mask]) )**2 ) )


print("\nFITTED PARAMETERS")
print("-----------------")
print("thetaR =", thetaR_fit)
print("thetaS =", thetaS_fit)
print("alpha  =", alpha_fit, "1/m")
print("n      =", n_fit)
print("b      =", b_fit)
print("lambda =", lam_fit)
print("Ks     =", ks_fit, "m/s")

print("\nRETENTION ERROR")
print("---------------")
print("RMSD dry   =", rmsd_dry, "m3/m3")
print("RMSD wet   =", rmsd_wet, "m3/m3")
print("RMSD total =", rmsd_ret, "m3/m3")

print("\nCONDUCTIVITY ERROR")
print("------------------")
print("RMSD log10(K) =", rmsd_k_log)























