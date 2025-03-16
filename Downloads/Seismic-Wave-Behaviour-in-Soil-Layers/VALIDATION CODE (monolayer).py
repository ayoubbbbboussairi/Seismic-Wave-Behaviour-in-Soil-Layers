
import numpy as np
import matplotlib.pyplot as plt

# 1. CURVE VALIDATION:

# Constants
theta1 = np.radians(0)
theta2 = np.radians(0)

V_s1 = 200  # Shear wave velocity of the surface layer (m/s)
h = 20  # Thickness of the surface layer (m)
rho1 = 2000  # Density of the surface layer (kg/m³)

# Shear wave velocity and density of the half-space (reference values)
V_s2 = np.array([1000, 600, 400], dtype=np.int64)
rho2 = np.array([2500, 2300, 2000])

# Frequency range
f = np.arange(0, 14, 0.01)

# Imaginary unit
j = complex(0, 1)

# Shear modulus of the surface layer
mu1 = rho1 * V_s1**2

# Angular frequency
w = 2 * np.pi * f

# Vertical component of the wave number
k_z1 = w * np.cos(theta1) / V_s1

# Transfer function for the infinite case
T_inf = 1 / np.cos(k_z1 * h)
T_infmod = np.abs(T_inf)

# Transfer function for different cases
T_mod = np.zeros((3, len(f)))

for i in range(3):
    # Shear modulus of the half-space
    mu2_i = rho2[i] * V_s2[i]**2

    # Factor X, accounting for impedance contrast
    X_i = np.sqrt((mu1 * rho1) / (mu2_i * rho2[i])) * (np.cos(theta1) / np.cos(theta2))

    # Transfer function for the given case
    T_i = 1 / (np.cos(k_z1 * h) + j * X_i * np.sin(k_z1 * h))
    T_mod[i, :] = np.abs(T_i)

# Plotting 
plt.plot(f, T_infmod, label=f'Infinite Case ($V_s = {V_s1}$m/s)')

for i in range(3):
    plt.plot(f, T_mod[i], label=f'$V_2 =$ {V_s2[i]}m/s')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Transfer Function')
plt.xlim(0, 14)
plt.ylim(0, 10)
plt.legend()
plt.show()