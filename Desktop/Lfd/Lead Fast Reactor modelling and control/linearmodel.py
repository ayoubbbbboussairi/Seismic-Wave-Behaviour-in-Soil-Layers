
import numpy as np
import matplotlib.pyplot as plt
import control  

# -------------------------------------------------------
# 1) Define numeric parameter values 
# -------------------------------------------------------
beta    = 319 * 1e-5     # total delayed neutron fraction (dimensionless)
beta1   = 6.142 * 1e-5
beta2   = 71.40 * 1e-5
beta3   = 34.86 * 1e-5
beta4   = 114.1 * 1e-5
beta5   = 69.92 * 1e-5
beta6   = 22.68 * 1e-5

Lambda  = 0.8066 * 1e-6    # neutron generation time [s] (example)
lambda1 = 0.0125       # decay constants [1/s] (example)
lambda2 = 0.0292
lambda3 = 0.0895
lambda4 = 0.2575
lambda5 = 0.6037
lambda6 = 2.6688

alpha_D_eoc = -0.17
alpha_C_eoc = -1.995
alpha_A_eoc = -0.2374
alpha_R_eoc = -0.7144

alpha_f = (alpha_D_eoc + alpha_A_eoc) * 1e-5
alpha_c = (alpha_C_eoc + alpha_R_eoc) * 1e-5
alpha_h = 1e-5        # reactivity coeff w.r.t control rod [pcm / rod_step] .

P0_k_tauf      = 373.25      #
tau_f   = 1.336       #
tau_c   = 1.321        #
tau_0   = 0.210         #

# -------------------------------------------------------
# 2) Construct A, B, C, D matrices
# -------------------------------------------------------
A = np.array([
    [-beta/Lambda,   beta1/Lambda,  beta2/Lambda,  beta3/Lambda,  beta4/Lambda,  beta5/Lambda,  beta6/Lambda,  alpha_f/Lambda,  alpha_c/Lambda],
    [lambda1,        -lambda1,      0,             0,             0,             0,             0,             0,               0             ],
    [lambda2,        0,             -lambda2,      0,             0,             0,             0,             0,               0             ],
    [lambda3,        0,             0,             -lambda3,      0,             0,             0,             0,               0             ],
    [lambda4,        0,             0,             0,             -lambda4,      0,             0,             0,               0             ],
    [lambda5,        0,             0,             0,             0,             -lambda5,      0,             0,               0             ],
    [lambda6,        0,             0,             0,             0,             0,             -lambda6,      0,               0             ],
    [P0_k_tauf,   0,             0,             0,             0,             0,             0,             -1/tau_f,        1/tau_f       ],
    [0,              0,             0,             0,             0,             0,             0,             1/tau_c,         -(2/tau_0 + 1/tau_c)]
], dtype=float)

B = np.array([
    [alpha_h/Lambda,  0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               0         ],
    [0,               2/tau_0   ]
], dtype=float)

# We have 11 outputs (9 states + T_out + reactivity)
C = np.array([
    [1,0,0,0,0,0,0,0,0],   # y1 = delta psi
    [0,1,0,0,0,0,0,0,0],   # y2 = delta eta1
    [0,0,1,0,0,0,0,0,0],   # y3 = delta eta2
    [0,0,0,1,0,0,0,0,0],   # y4 = delta eta3
    [0,0,0,0,1,0,0,0,0],   # y5 = delta eta4
    [0,0,0,0,0,1,0,0,0],   # y6 = delta eta5
    [0,0,0,0,0,0,1,0,0],   # y7 = delta eta6
    [0,0,0,0,0,0,0,1,0],   # y8 = delta T_f
    [0,0,0,0,0,0,0,0,1],   # y9 = delta T_c
    [0,0,0,0,0,0,0,0,2],   # y10= 2 delta T_c -> partial for T_out
    [0,0,0,0,0,0,0,alpha_f, alpha_c]  # y11= alpha_f*T_f + alpha_c*T_c -> partial for delta rho
], dtype=float)

D = np.array([
    [0, 0],   # y1
    [0, 0],   # y2
    [0, 0],   # y3
    [0, 0],   # y4
    [0, 0],   # y5
    [0, 0],   # y6
    [0, 0],   # y7
    [0, 0],   # y8
    [0, -1],  # y9 -> 
    [0, 0],   # actually for row 10 we might fix in code
    [alpha_h, 0]  # y11 -> alpha_h * dh + ...
], dtype=float)



D[9,1] = -1.0  # y10 (the 10th row) direct feedthrough from - delta T_in

sys = control.ss(A, B, C, D)

# Simulation parameters
t_final = 50.0
n_points = 501
t = np.linspace(0, t_final, n_points)

# Two scenarios: (A) dh=1 pcm step, (B) dTin=+20 °C
uA = np.zeros((2, n_points))  # shape = (n_inputs, len(t))
uB = np.zeros((2, n_points))

# Scenario A
uA[0, :] = 10 * 1e-5   # dh = 1 pcm
uA[1, :] = 0.0   # dTin = 0

# Scenario B
uB[0, :] = 0.0   # dh = 0
uB[1, :] = 20.0  # dTin = +20 C


respA = control.forced_response(sys, T=t, U=uA, return_x=True)
respB = control.forced_response(sys, T=t, U=uB, return_x=True)

# Extract results
tA, yA, xA = respA.time, respA.outputs, respA.states
tB, yB, xB = respB.time, respB.outputs, respB.states

#   row 10 is delta T_out => index 9.
ipower= 0
ieta1 = 1
ieta2 = 2
ieta3 = 3
ieta4 = 4
ieta5= 5
ieta6 = 6
iTf  = 7  #
iTc = 8
iTout = 9
irho=10

# Figure 1 - Scenario A
plt.figure(figsize=(15, 10))
titles = [' psi', ' eta1', ' eta2', ' eta3', ' eta4', ' eta5', ' eta6', 'Tf (C)', 'Tc (C)', ' Tout (c)', 'total reactivity']
indices = [ipower, ieta1, ieta2, ieta3, ieta4, ieta5, ieta6, iTf, iTc, iTout, irho]

for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.plot(tA, yA[indices[i], :])
    plt.xlabel('Time [s]')
    plt.ylabel(titles[i])
    plt.title(f'Scenario A: dh=1 pcm - {titles[i]}')
plt.tight_layout()

# Figure 2 - Scenario B
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.plot(tB, yB[indices[i], :], 'r')
    plt.xlabel('Time [s]')
    plt.ylabel(titles[i])
    plt.title(f'Scenario B: dTin=+20 °C - {titles[i]}')
plt.tight_layout()
plt.show()