# -*- coding: utf-8 -*-
"""Regenerate Table 4: linear RMSD between exact and approximate relative conductivity."""

import csv
from pathlib import Path
import numpy as np
from scipy.integrate import quad

# Rosetta parameters: alpha [1/m], n [-]. Other parameters cancel from Krel.
soils = {"Loamy sand": (3.475361614, 1.745822153), "Silt loam": (0.505824662, 1.663412650), "Silty clay": (1.621810097, 1.321295634)}
b_values = [0.1, 0.5, 1.0, 2.0]
lam = 0.5
h_min = 1e-3  # m
h_max = 1e3  # m
n_points = 1000
h = np.logspace(np.log10(h_min), np.log10(h_max), n_points)


def saturation(h, alpha, n, b):
    m_b = 1.0 - 1.0 / n + b
    return np.exp(-m_b * np.logaddexp(0.0, n * np.log(alpha * h)))


# Eq. 49, evaluated without subtracting nearly equal numbers in the dry tail.
def krel_approx(Se, n, b, lam):
    m_b = 1.0 - 1.0 / n + b
    z = np.exp(np.log(Se) / m_b)
    bracket = -np.expm1((1.0 - 1.0 / n) * np.log1p(-z))
    return Se**lam * bracket**2


# Inverse of the head H(Se), with stable evaluation near Se=1.
def inv_h(Se, alpha, n, b):
    m_b = 1.0 - 1.0 / n + b
    log_z = np.log(Se) / m_b
    return alpha * np.exp((log_z - np.log(-np.expm1(log_z))) / n)


# Exact Mualem integral. Compute its common denominator once per soil and b.
def krel_exact(Se, alpha, n, b, lam):
    denominator = quad(inv_h, 0.0, 1.0, args=(alpha, n, b), epsabs=1e-12, epsrel=1e-10, limit=500)[0]
    numerator = np.array([s * quad(lambda u: inv_h(s * u, alpha, n, b), 0.0, 1.0, epsabs=1e-12, epsrel=1e-10, limit=500)[0] for s in Se])
    return Se**lam * (numerator / denominator)**2


def evaluate():
    rows = []
    for soil, (alpha, n) in soils.items():
        for b in b_values:
            Se = saturation(h, alpha, n, b)
            exact = krel_exact(Se, alpha, n, b, lam)
            approx = krel_approx(Se, n, b, lam)
            error = np.sqrt(np.mean((approx - exact)**2))
            rows.append((soil, b, error))
    return rows


if __name__ == "__main__":
    rows = evaluate()
    print(f"Linear Krel RMSD; lambda={lam:g}; {n_points} heads logarithmically spaced from {h_min:g} to {h_max:g} m.")
    print(f"{'Soil':<14} {'b':>5} {'Krel RMSD':>12}")
    for soil, b, error in rows:
        print(f"{soil:<14} {b:5.1f} {error:12.8f}")
    errors = np.array([row[2] for row in rows])
    print(f"Mean RMSD across cases = {np.mean(errors):.8f}")
    print(f"Maximum RMSD = {np.max(errors):.8f}")
    output_file = Path(__file__).resolve().parent / "table4_conductivity_rmsd.csv"
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["soil", "b", "Krel_RMSD"])
        writer.writerows(rows)
    print(f"Saved: {output_file}")
