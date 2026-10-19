# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt




save_ = False

soils = {
    "Loamy sand": (0.049, 0.390, 3.475361614, 1.745822153, 1.21755E-05),
    "Silt loam":  (0.065, 0.439, 0.505824662, 1.663412650, 2.11099E-06),
    "Silty clay": (0.111, 0.481, 1.621810097, 1.321295634, 1.11298E-06)
}

b_values = [0.1, 0.5, 1.0, 2.0]

#b_values = [0.01, 0.1]



h = np.logspace(-3, 3, 1000)


def saturation(h, alpha, n, b):
    return (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n + b))


def integrand(x, alpha, n, b):
    m = 1.0 - 1.0 / n

    C = m * n * alpha**n * x**(n - 1.0) \
        * (1.0 + (alpha * x)**n)**(-m - 1.0)

    return C * np.exp(-b * alpha * x)


def saturation_integral(h, alpha, n, b):
    Z = quad(integrand, 0.0, np.inf, args=(alpha, n, b))[0]

    return quad(
        integrand, h, np.inf,
        args=(alpha, n, b)
    )[0] / Z


def saturation_vg(h, alpha, n):
    return (1.0 + (alpha * h)**n)**(-(1.0 - 1.0/n))


def rmsd(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))


for soil, parameters in soils.items():

    thetaR, thetaS, alpha, n, ks = parameters

    theta_vg = saturation_vg(h, alpha, n)

    #plt.figure(figsize=(5, 4.5))
    plt.figure(figsize=(4, 3.5))

    plt.plot(
        h, theta_vg,
        linewidth=2.0,
        label="VG, b = 0"
    )

    for b in b_values:

        theta_approx = saturation(h, alpha, n, b)

        theta_integral = np.array([
            saturation_integral(x, alpha, n, b)
            for x in h
        ])

        error = rmsd(theta_approx, theta_integral)

        #print(f"{soil:12s}, b = {b:3.1f}, RMSD = {error:.6f}")
        print(f"{soil},{b:3.1f},{error:.6f}")

        line = plt.plot(
            h,
            theta_approx,
            label=f"b = {b}"
        )[0]

        plt.plot(
            h,
            theta_integral,
            "--",
            color=line.get_color(),
            label=f"b = {b}"
        )

    plt.xscale("log")

    plt.xlabel("Pressure head, h [m]")
    plt.ylabel("Saturation, Θ [-]")

    plt.title(soil)

    plt.ylim(0, 1.0)
    plt.xlim(1e-2, 1e3)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    file_name = soil.lower().replace(" ", "_") + ".png"
    
    if save_:
        plt.savefig(rf"C:\Users\L\Desktop\{file_name}", dpi=600, bbox_inches="tight")

    plt.show()







































