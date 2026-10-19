# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


save_ = False

FIG_W = 4
FIG_H = 3.5


# Brooks-Corey parameters
# Vereecken et al. (2007), loamy sand
theta_r = 0.057
theta_s = 0.410
alpha = 17.7       # m^-1
eta = 1.07


def theta_BC(h, b):
    h = np.asarray(h, dtype=float)

    Se = np.ones_like(h)

    mask = h > 1.0 / alpha

    if b == 0:
        Se[mask] = (alpha * h[mask]) ** (-eta)

    else:
        gamma_den = mp.gammainc(-eta, b, mp.inf)

        Se[mask] = [
            float(
                mp.gammainc(-eta, b * alpha * hi, mp.inf)
                / gamma_den
            )
            for hi in h[mask]
        ]

    return theta_r + (theta_s - theta_r) * Se


h = np.logspace(-3, 3, 1000)

b_values = [0.0, 0.1, 0.5, 1.0]


plt.figure(figsize=(FIG_W, FIG_H))

for b in b_values:
    theta = theta_BC(h, b)
    plt.semilogx(h, theta, label=f"b = {b}")


plt.xlabel("Pressure head, h [m]")
plt.ylabel(r"Water content, $\theta$ [m$^3$ m$^{-3}$]")

plt.legend()
plt.grid()

plt.tight_layout()

if save_:
    plt.savefig(r"C:\Users\L\Desktop\brooks.pdf", bbox_inches="tight")
    plt.savefig(r"C:\Users\L\Desktop\brooks.png", dpi=600, bbox_inches="tight")
    plt.savefig(r"C:\Users\L\Desktop\brooks.svg",bbox_inches="tight")

plt.show()