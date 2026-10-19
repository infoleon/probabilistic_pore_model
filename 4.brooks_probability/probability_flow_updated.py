# -*- coding: utf-8 -*-
"""Back-transformed coefficients and probabilities consistent with the BTC fit."""
import numpy as np
from scipy.integrate import quad

# Same inputs as the transport calibration; heads are positive magnitudes.
theta_r = 0.165
theta_s = 0.387
alpha = 4.20  # 1/m
n = 4.44
h_max = 1e3  # m


h_a = 0.231
h_b = 0.241
b_102 = 1.422  # selected local solution
b_263 = 2.050
theta_102 = 0.221
theta_263 = 0.2475



def effective_saturation(theta):
    return (theta - theta_r) / (theta_s - theta_r)


def h_from_saturation(Theta, b):
    m_b = 1.0 - 1.0 / n + b
    return np.expm1(-np.log(Theta) / m_b)**(1.0 / n) / alpha


# Pointwise back-transformation; the coefficient depends on h.
def b_phys(h, b):
    return b * np.log1p((alpha * h)**n) / (alpha * h)


# Full-domain density, normalized over 0 < h < infinity.
def P_VG(h, b):
    m_b = 1.0 - 1.0 / n + b
    z = n * np.log(alpha * h)
    return np.exp(np.log(n * m_b) + z - np.log(h) - (m_b + 1.0) * np.logaddexp(0.0, z))


def probability_interval(h_lower, h_upper, b):
    return quad(lambda x: P_VG(np.exp(x), b) * np.exp(x), np.log(h_lower), np.log(h_upper), epsabs=1e-12, epsrel=1e-10)[0]


# Conditional water-volume density; this is not a flux-weighted density.
def p_transport(h, b, h_min, M0):
    return P_VG(h, b) / M0 if h_min <= h <= h_max else 0.0


def transport_probability_interval(h_lower, h_upper, b, h_min, M0):
    lower = max(h_lower, h_min)
    upper = min(h_upper, h_max)
    return probability_interval(lower, upper, b) / M0 if upper > lower else 0.0


def report_flow(q_cm_h, theta, b):
    Theta = effective_saturation(theta)
    h_min = h_from_saturation(Theta, b)
    M0 = probability_interval(h_min, h_max, b)
    p_full = probability_interval(h_a, h_b, b)
    p_tr = transport_probability_interval(h_a, h_b, b, h_min, M0)
    print(f"q = {q_cm_h:.2f} cm/h")
    print(f"theta = {theta:.6f}, Theta = {Theta:.6f}, b = {b:.8f}")
    print(f"h_min = {h_min:.8f} m, h_max = {h_max:g} m")
    print(f"b_phys(h_min) = {b_phys(h_min, b):.8f}")
    print(f"P_VG(h_min) = {P_VG(h_min, b):.8f} m^-1")
    print(f"p_transport(h_min) = {p_transport(h_min, b, h_min, M0):.8f} m^-1")
    print(f"Full-domain P({h_a:g} <= h <= {h_b:g} m) = {p_full:.8f}")
    print(f"Conditional transport P({h_a:g} <= h <= {h_b:g} m) = {p_tr:.8f}")
    print()


if __name__ == "__main__":
    report_flow(1.02, theta_102, b_102)
    report_flow(2.63, theta_263, b_263)
