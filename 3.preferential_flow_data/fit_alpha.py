# -*- coding: utf-8 -*-

"""
Fit b from Zurmuhl wetting/drying retention curves
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

###############################
######## PARAMETERS ###########
###############################

thetaR = 0.165
thetaS = 0.387

alpha_d = 4.20  # 1/m, drying
alpha_w = 5.68  # 1/m, wetting

n = 4.44
m = 1.0 - 1.0/n

###############################
########## FUNCTIONS ##########
###############################

# Traditional VG retention
def retention_vg(h, thetaR, thetaS, alpha, n):
    m = 1.0 - 1.0/n
    Se = (1.0 + (alpha * h)**n)**(-m)
    return thetaR + (thetaS - thetaR) * Se

# Modified retention with b
def retention_b(h, thetaR, thetaS, alpha, n, b):
    m = 1.0 - 1.0/n
    Se = (1.0 + (alpha * h)**n)**(-(m + b))
    return thetaR + (thetaS - thetaR) * Se

def residuals_b(par, h, theta_target, thetaR, thetaS, alpha, n):
    b = par[0]
    theta_pred = retention_b(h, thetaR, thetaS, alpha, n, b)
    return theta_pred - theta_target

###############################
######## SAMPLE DATA ##########
###############################

# Generate synthetic "measured" wetting data from Zurmuhl's alpha_w
h_data = np.logspace(-3, 3, 30)
theta_wet = retention_vg(h_data, thetaR, thetaS, alpha_w, n)

###############################
############ FIT ##############
###############################

fit = least_squares(residuals_b, x0=[0.1], bounds=(0.0, 10.0),
                    args=(h_data, theta_wet, thetaR, thetaS, alpha_d, n))

b_fit = fit.x[0]

###############################
######## CALCULATE FIT ########
###############################

theta_fit = retention_b(h_data, thetaR, thetaS, alpha_d, n, b_fit)
rmsd = np.sqrt(np.mean((theta_fit - theta_wet)**2))

###############################
########### PRINT #############
###############################

print("b =", b_fit)
print("RMSD =", rmsd)

###############################
############ PLOT #############
###############################

# Smooth domain only for drawing curves
h_plot = np.logspace(-3, 3, 1000)

# Original drying curve
theta_dry_plot = retention_vg(h_plot, thetaR, thetaS, alpha_d, n)

# Original wetting curve
theta_wet_plot = retention_vg(h_plot, thetaR, thetaS, alpha_w, n)

# Modified curve using alpha_d + fitted b
theta_b_plot = retention_b(h_plot, thetaR, thetaS, alpha_d, n, b_fit)

plt.figure()

# Synthetic wetting points
plt.scatter(h_data, theta_wet, label="Wetting data (sampled)")

# Traditional drying VG
plt.plot(h_plot, theta_dry_plot, label="VG drying")

# Modified model
plt.plot(h_plot, theta_b_plot, label=f"Modified VG, b = {b_fit:.4f}")

# Traditional wetting VG
plt.plot(h_plot, theta_wet_plot, linestyle="--", label="VG wetting")

plt.xscale("log")
plt.xlabel("Pressure head h [m]")
plt.ylabel("Water content θ [m³/m³]")

plt.legend()
plt.tight_layout()
plt.show()