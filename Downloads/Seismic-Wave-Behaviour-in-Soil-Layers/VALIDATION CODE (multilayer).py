
import numpy as np
import matplotlib.pyplot as plt

# 1. CURVE VALIDATION:

# Constants
theta1 = np.radians(0)
theta2 = np.radians(0)
thetaBR = np.radians(0)

V_s1 = np.array([150, 100, 75], dtype=np.int64)  # Shear wave velocity of layer 1 (m/s)
h1 = 10  # Thickness of layer 1 (m)
rho1 = 2300  # Density of layer 1 (kg/m³)

V_s2 = np.array([250, 300, 325], dtype=np.int64)  # Shear wave velocity of layer 2 (m/s)
h2 = 10  # Thickness of layer 2 (m)
rho2 = 2400  # Density of layer 2 (kg/m³)

V_sBR = 1000  # Shear wave velocity of the bedrock (m/s)
rhoBR = 2500  # Density of the bedrock (kg/m³)

# Frequency range
f = np.arange(0.01, 14, 0.01)

# Initial amplitude factor
A_1 = 1

# Empty arrays for calculations
k_z1 = np.empty((3, len(f)), dtype=np.complex128)
k_z2 = np.empty((3, len(f)), dtype=np.complex128)
A_2 = np.empty((3, len(f)), dtype=np.complex128)
A_dash2 = np.empty((3, len(f)), dtype=np.complex128)
A_3 = np.empty((3, len(f)), dtype=np.complex128)
A_dash3 = np.empty((3, len(f)), dtype=np.complex128)
T_1 = np.empty((3, len(f)), dtype=np.complex128)
T_2 = np.empty((3, len(f)), dtype=np.complex128)
T_tot = np.empty((3, len(f)), dtype=np.complex128)
T = np.empty((3, len(f)), dtype=np.float64)

# Imaginary number
j = complex(0, 1)

# Shear modulus calculations
mu1 = rho1 * V_s1**2
mu2 = rho2 * V_s2**2
muBR = rhoBR * V_sBR**2

# Angular frequency
w = 2 * np.pi * f

# Vertical component of the wave number for each layer
for i in range(3):
    k_z1[i] = w * np.cos(theta1) / V_s1[i]
    k_z2[i] = w * np.cos(theta2) / V_s2[i]

k_zBR = w * np.cos(thetaBR) / V_sBR

# X factors for impedance contrast
X1 = np.sqrt(mu1 * rho1 / (mu2 * rho2)) * (np.cos(theta1) / np.cos(theta2))
X2 = np.sqrt(mu2 * rho2 / (muBR * rhoBR)) * (np.cos(theta2) / np.cos(thetaBR))

# A factors of the transfer functions
for i in range(3):
    A_2[i] = (1/2) * A_1 * (1 + X1[i]) * np.exp(-j * k_z1[i] * h1) + \
             (1/2) * A_1 * (1 - X1[i]) * np.exp(j * k_z1[i] * h1)

    A_dash2[i] = (1/2) * A_1 * (1 - X1[i]) * np.exp(-j * k_z1[i] * h1) + \
                 (1/2) * A_1 * (1 + X1[i]) * np.exp(j * k_z1[i] * h1)

# A factors for layer 3
for i in range(3):
    A_3[i] = (1/2) * A_2[i] * (1 + X2[i]) * np.exp(-j * k_z2[i] * h2) + \
             (1/2) * A_dash2[i] * (1 - X2[i]) * np.exp(j * k_z2[i] * h2)

    A_dash3[i] = A_3[i]  # Assuming A_dash3 = A_3 for simplified implementation

# Local transfer functions
for i in range(3):
    T_1[i] = (2 * A_1) / (A_2[i] + A_dash2[i])
    T_2[i] = (A_2[i] + A_dash2[i]) / (A_3[i] + A_dash3[i])

# Total transfer function
for i in range(3):
    T_tot[i] = T_1[i] * T_2[i]

# Compute absolute values of transfer function
for i in range(3):
    T[i] = np.abs(T_tot[i])

for i in range(3):
    plt.plot(f, T[i], label=f'$V_1 = {V_s1[i]}$ m/s, $V_2 = {V_s2[i]}$ m/s')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Transfer Function')
plt.xlim(0, 14)
plt.ylim(0, 15)
plt.legend()
plt.show()