# -*- coding: utf-8 -*-

"""

Fit retention


"""


import re
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



h_dry_file = r"./experiment_data_ret_k_fit/1300_h_dry_.txt"
h_wet_file = r"./experiment_data_ret_k_fit/1300_h_wet_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/2002_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/2002_h_wet_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/2020_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/2020_h_wet_.txt"

#h_dry_file = r"./experiment_data_ret_k_fit/4870_h_dry_.txt"
#h_wet_file = r"./experiment_data_ret_k_fit/4870_h_wet_.txt"


# soil data here
#                              theta_r, theta_s,  alpha,       n,             ks 
#loamy sand
thetaR, thetaS, alpha, n, ks = 0.049,    0.39,    3.475361614, 1.745822153,   1.21755E-05
#silty loam
thetaR, thetaS, alpha, n, ks = 0.065,    0.439,   0.505824662, 1.66341265,    2.11099E-06
#silty clay
#thetaR, thetaS, alpha, n, ks = 0.111,    0.481,   1.621810097, 1.321295634,   1.11298E-06


b = 0.01


# Domain
ini = 0.001
end = 1000.0

N = int(1000)
# domain of h
h = np.logspace(np.log10(ini), np.log10(end), 1000)

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

def retention(h, thetaR, thetaS, alpha, n, b):
    Se = (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n + b))
    return thetaR + (thetaS - thetaR) * Se


def residuals(par, dry_dat, wet_dat):
    thetaR, thetaS, alpha, n, b = par

    theta_dry = dry_dat[:, 0]
    h_dry = dry_dat[:, 1]

    theta_wet = wet_dat[:, 0]
    h_wet = wet_dat[:, 1]

    pred_dry = retention(h_dry, thetaR, thetaS, alpha, n, 0.0)
    pred_wet = retention(h_wet, thetaR, thetaS, alpha, n, b)

    res_dry = pred_dry - theta_dry
    res_wet = pred_wet - theta_wet

    return np.concatenate((res_dry, res_wet))


######################################
########## end of FUNCTIONS ##########
######################################



dry_dat = read_dat(h_dry_file)
wet_dat = read_dat(h_wet_file)




# fitting

x0 = [thetaR, thetaS, alpha, n, b]

lower = [0.0, 0.0, 1e-8, 1.000001, 0.0]
upper = [1.0, 1.0, np.inf, np.inf, np.inf]

fit = least_squares(
    residuals,
    x0,
    bounds=(lower, upper),
    args=(dry_dat, wet_dat)
)

thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit = fit.x


# calculating

theta_dry_obs = dry_dat[:, 0]
h_dry = dry_dat[:, 1]

theta_wet_obs = wet_dat[:, 0]
h_wet = wet_dat[:, 1]

theta_dry_pred = retention(
    h_dry,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, 0.0
)

theta_wet_pred = retention(
    h_wet,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit
)


# error

def rmsd(obs, pred):
    return np.sqrt(np.mean((obs - pred)**2))


rmsd_dry = rmsd(theta_dry_obs, theta_dry_pred)
rmsd_wet = rmsd(theta_wet_obs, theta_wet_pred)

rmsd_total = np.sqrt(
    np.mean(
        np.concatenate((
            theta_dry_pred - theta_dry_obs,
            theta_wet_pred - theta_wet_obs
        ))**2
    )
)


# printint the error

print("thetaR =", thetaR_fit)
print("thetaS =", thetaS_fit)
print("alpha  =", alpha_fit)
print("n      =", n_fit)
print("b      =", b_fit)

print("RMSD dry   =", rmsd_dry)
print("RMSD wet   =", rmsd_wet)
print("RMSD total =", rmsd_total)



# plotting data

h_plot = np.logspace(np.log10(1e-3), np.log10(1.7e3), 1000)

theta_dry_curve = retention(
    h_plot,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, 0.0
)

theta_wet_curve = retention(
    h_plot,
    thetaR_fit, thetaS_fit, alpha_fit, n_fit, b_fit
)

# plotting

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



















