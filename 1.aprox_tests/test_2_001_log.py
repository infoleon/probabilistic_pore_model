import numpy as np
from scipy.integrate import quad

# ============================================================
# SOIL DATA
# theta_r, theta_s, alpha, n, Ks
# ============================================================

soils = {
    "Loamy sand": [0.049, 0.390, 3.475361614, 1.745822153, 1.21755e-5],
    "Silt loam":  [0.065, 0.439, 0.505824662, 1.663412650, 2.11099e-6],
    "Silty clay": [0.111, 0.481, 1.621810097, 1.321295634, 1.11298e-6]
}

lam = 0.5
b_values = [0.1, 0.5, 1.0, 2.0]

# pressure-head domain used in the manuscript
h = np.logspace(-3, 3, 1000)

# ============================================================
# FUNCTIONS
# ============================================================

# saturation with analytical approximation
def saturation(h, alpha, n, b):
    m = 1.0 - 1.0 / n + b
    return (1.0 + (alpha * h)**n)**(-m)

# numerically stable analytical Krel approximation
def krel_approx(theta, n, b, lam):
    q = n / (n * (1.0 + b) - 1.0)
    p = (n - 1.0) / n
    x = theta**q
    bracket = -np.expm1(p * np.log1p(-x))
    return theta**lam * bracket**2

# 1 / h(theta)
def inv_h(theta, alpha, n, b):
    q = n / (n * (1.0 + b) - 1.0)
    return alpha * theta**(q / n) * (1.0 - theta**q)**(-1.0 / n)

# numerical Mualem integration
def krel_integral(theta, alpha, n, b, lam):
    numerator = quad(inv_h, 0.0, theta, args=(alpha, n, b))[0]
    denominator = quad(inv_h, 0.0, 1.0, args=(alpha, n, b))[0]
    return theta**lam * (numerator / denominator)**2

# linear RMSD
def rmsd(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))

# logarithmic RMSD
def rmsd_log10(y1, y2):
    return np.sqrt(np.mean((np.log10(y1) - np.log10(y2))**2))

# ============================================================
# CALCULATIONS
# ============================================================

for soil_name, soil_data in soils.items():
    thetaR, thetaS, alpha, n, ks = soil_data

    print("\n" + "=" * 60)
    print(soil_name)
    print("=" * 60)

    for b in b_values:
        theta = saturation(h, alpha, n, b)
        k_approx = krel_approx(theta, n, b, lam)
        k_integral = np.array([krel_integral(x, alpha, n, b, lam) for x in theta])

        error_linear = rmsd(k_approx, k_integral)
        error_log = rmsd_log10(k_approx, k_integral)
        max_log_error = np.max(np.abs(np.log10(k_approx) - np.log10(k_integral)))

        rms_factor = 10**error_log
        max_factor = 10**max_log_error

        print(f"b = {b:3.1f} | Krel RMSD = {error_linear:.6f} | log10(Krel) RMSD = {error_log:.6f} | RMS factor = {rms_factor:.2f} | max factor = {max_factor:.2f}")