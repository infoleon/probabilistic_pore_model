# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


save_ = False

FIG_W = 4
FIG_H = 3.5
DPI = 600

# Zurmuhl 1998 hydraulic parameters
thetaR = 0.165
thetaS = 0.387
alpha = 4.20   # 1/m
n = 4.44

h_min_plot = 1e-4   # m not h_mion on the article, actually, the plot h min
h_max_plot = 1e3    # m

# choose one fitted b value
b = 1.422
# b = 1.2191453634519802

N = 1.0   # use N=1 to plot the normalized shape


def theta_eff(h, alpha, n, b):
    q = 1.0 + b - 1.0 / n
    return (1.0 + (alpha * h) ** n) ** (-q)


def socc(h, alpha, n, b, N=1.0):
    th = theta_eff(h, alpha, n, b)
    th = np.clip(th, 1e-12, 1.0 - 1e-12)
    return -N * (th * np.log(th) + (1.0 - th) * np.log(1.0 - th))


def h_star(alpha, n, b):
    q = 1.0 + b - 1.0 / n
    return ((2.0 ** (1.0 / q) - 1.0) ** (1.0 / n)) / alpha


h = np.logspace(np.log10(h_min_plot), np.log10(h_max_plot), 2000)
S = socc(h, alpha, n, b, N)

hmax_S = h_star(alpha, n, b)
Smax = socc(np.array([hmax_S]), alpha, n, b, N)[0]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

ax.plot(h, S, label=f"b = {b:.3f}")
ax.axvline(hmax_S, linestyle="--", linewidth=0.8, label=fr"$h^* = {hmax_S:.3f}$ m")
#ax.plot(hmax_S, Smax, marker="o", linestyle="None")

ax.set_xlim(0.01, 10.0)

ax.set_xscale("log")
ax.set_xlabel("Pressure head, h [m]")
ax.set_ylabel(r"$S_{occ}$ [-]")
ax.set_title("Configurational entropy of filled/unfilled pores")
ax.grid(True, which="both", linewidth=0.4)
ax.legend(frameon=False)

fig.tight_layout()

out_dir = Path(__file__).resolve().parent
base = out_dir / "fig_socc_zurmuhl"

if save_:
    fig.savefig(str(base) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")

plt.show()

print(f"b = {b:.12f}")
print(f"h* = {hmax_S:.6f} m")
print(f"Socc_max = {Smax:.6f}")









