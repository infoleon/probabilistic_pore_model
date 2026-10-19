# -*- coding: utf-8 -*-
"""Fit BTCs with conditional normalization and a b-dependent lower head bound."""

import re
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



save_ = True
FIG_W = 4
FIG_H = 3.5
s = 18
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})


c_c0_f_102 = r"./btc_1.02.txt"
c_c0_f_263 = r"./btc_2.63.txt"


# Zurmuhl hydraulic parameters; pressure heads are positive magnitudes.
thetaR = 0.165
thetaS = 0.387
alpha = 4.20  # 1/m, drying reference
n = 4.44
L = 0.145  # m
h_max = 1e3  # m, described finite upper head cutoff


# Flux densities, water contents and mean water velocities.
q_102 = 1.02e-2 / 3600.0  # m/s
q_263 = 2.63e-2 / 3600.0  # m/s
theta_102 = 0.221
theta_263 = 0.2475
v_102 = q_102 / theta_102
v_263 = q_263 / theta_263



def read_dat_c(file):
    dat = []
    with open(file, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                tempo, cc0 = re.split(r"[,\t]", line)
                dat.append([float(tempo), float(cc0)])
    return np.array(dat)


def h_from_theta(theta, thetaR, thetaS, alpha, n, b):
    Se = (theta - thetaR) / (thetaS - thetaR)
    m_b = 1.0 - 1.0 / n + b
    return np.expm1(-np.log(Se) / m_b)**(1.0 / n) / alpha


# P_VG(h,b) dh in x = ln(h), including the Jacobian dh = h dx.
def p_integrand_log(x, alpha, n, b):
    m_b = 1.0 - 1.0 / n + b
    z = n * (np.log(alpha) + x)
    return np.exp(np.log(n * m_b) + z - (m_b + 1.0) * np.logaddexp(0.0, z))


# h^-2 P_VG(h,b) dh in x = ln(h).
def transport_integrand_log(x, alpha, n, b):
    m_b = 1.0 - 1.0 / n + b
    z = n * (np.log(alpha) + x)
    return np.exp(np.log(n * m_b) + z - 2.0 * x - (m_b + 1.0) * np.logaddexp(0.0, z))



# Eq. 73: A = (q/theta) M0/M2, with h_min recalculated for every b.
def transport_state(alpha, n, b, v_mean, theta, thetaR, thetaS, h_max):
    h_min = h_from_theta(theta, thetaR, thetaS, alpha, n, b)
    x_min = np.log(h_min)
    x_max = np.log(h_max)
    M0 = quad(p_integrand_log, x_min, x_max, args=(alpha, n, b), epsabs=1e-12, epsrel=1e-10, limit=500)[0]
    M2 = quad(transport_integrand_log, x_min, x_max, args=(alpha, n, b), epsabs=1e-12, epsrel=1e-10, limit=500)[0]
    A = v_mean * M0 / M2
    return h_min, M0, M2, A


def calc_A(alpha, n, b, v_mean, theta, thetaR, thetaS, h_max):
    return transport_state(alpha, n, b, v_mean, theta, thetaR, thetaS, h_max)[3]


# Eqs. 77-80: flux-weighted step-input BTC, including both arrival limits.
def btc_model(t, alpha, n, b, v_mean, L, theta, thetaR, thetaS, h_max):
    h_min, M0, M2, A = transport_state(alpha, n, b, v_mean, theta, thetaR, thetaS, h_max)
    x_min = np.log(h_min)
    t_min = L * h_min**2 / A
    t_max = L * h_max**2 / A
    result = []
    for ti in np.atleast_1d(t):
        if ti <= t_min:
            cc0 = 0.0
        elif ti >= t_max:
            cc0 = 1.0
        else:
            h_t = np.sqrt(A * ti / L)
            numerator = quad(transport_integrand_log, x_min, np.log(h_t), args=(alpha, n, b), epsabs=1e-12, epsrel=1e-10, limit=500)[0]
            cc0 = numerator / M2
        result.append(cc0)
    return np.array(result)


def residuals_b(par, t_obs, cc0_obs, alpha, n, v_mean, L, theta, thetaR, thetaS, h_max):
    return btc_model(t_obs, alpha, n, par[0], v_mean, L, theta, thetaR, thetaS, h_max) - cc0_obs


def fit_btc(t_obs, cc0_obs, v_mean, theta):
    fit = least_squares(residuals_b, x0=[1.2], bounds=(0.0, 10.0), args=(t_obs, cc0_obs, alpha, n, v_mean, L, theta, thetaR, thetaS, h_max))
    b = fit.x[0]
    pred = btc_model(t_obs, alpha, n, b, v_mean, L, theta, thetaR, thetaS, h_max)
    rmsd = np.sqrt(np.mean((pred - cc0_obs)**2))
    h_min, M0, M2, A = transport_state(alpha, n, b, v_mean, theta, thetaR, thetaS, h_max)
    return fit, b, pred, rmsd, h_min, A


def plot_btc(t_obs, cc0_obs, b, v_mean, theta, flux_label, output_stem):
    t_plot = np.linspace(0.0, max(t_obs) * 1.1, 300)
    pred_plot = btc_model(t_plot, alpha, n, b, v_mean, L, theta, thetaR, thetaS, h_max)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(t_obs, cc0_obs, s=s, label="Measured")
    ax.plot(t_plot, pred_plot, label=f"Model, b={b:.3f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("C/C0")
    ax.set_title(f"BTC - q = {flux_label} cm/h")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save_:
        fig.savefig(f"{output_stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    dat_102 = read_dat_c(c_c0_f_102)
    dat_263 = read_dat_c(c_c0_f_263)
    t_102, cc0_102 = dat_102[:, 0], dat_102[:, 1]
    t_263, cc0_263 = dat_263[:, 0], dat_263[:, 1]
    
    fit_102, b_102, pred_102, rmsd_102, h_min_102, A_102 = fit_btc(t_102, cc0_102, v_102, theta_102)
    fit_263, b_263, pred_263, rmsd_263, h_min_263, A_263 = fit_btc(t_263, cc0_263, v_263, theta_263)
    
    print(f"q = 1.02 cm/h: b = {b_102:.8g}, RMSD = {rmsd_102:.8g}, h_min = {h_min_102:.8g} m, A = {A_102:.8g} m^3/s")
    print(f"q = 2.63 cm/h: b = {b_263:.8g}, RMSD = {rmsd_263:.8g}, h_min = {h_min_263:.8g} m, A = {A_263:.8g} m^3/s")
    print(f"Optimizer 1.02: {fit_102.message}")
    print(f"Optimizer 2.63: {fit_263.message}")
    
    plot_btc(t_102, cc0_102, b_102, v_102, theta_102, "1.02", "btc_102")
    plot_btc(t_263, cc0_263, b_263, v_263, theta_263, "2.63", "btc_263")








