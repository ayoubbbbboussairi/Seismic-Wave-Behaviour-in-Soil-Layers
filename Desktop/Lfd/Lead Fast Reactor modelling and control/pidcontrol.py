
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def reactor_odes(t, y, params):
    """
    ODE system for the lead fast reactor with PID-based control rod reactivity.

    State vector y = [P, C, T_f, T_c, e, E].
    """
    (rho_0, alpha_h, alpha_f, alpha_c, beta, Lambda, lambd,
     M_f, C_f, M_c, C_c, K, G, T_in, T_f0, T_c0,
     K_p, K_i, Max_reactivity, ramp_slope) = params

    P, C, T_f, T_c, e, E = y

    # ---------------------------------------------------------
    # 1) Define the time-varying power setpoint: P_set(t)
    #    Suppose the initial power is P0, and we want a ramp
    #    of "ramp_slope" MW per minute. 
    # ---------------------------------------------------------
    # assume the initial power is y0[0] = P0
    # at t=0. Then:
    P0 = y0_initial[0]
    P_set = P0 + ramp_slope * t  # 1 MW/min ramp

    # ---------------------------------------------------------
    # 2) Error dynamics
    #    e(t) = P_set(t) - P(t)
    #    de/dt = dP_set/dt - dP/dt = ramp_slope - dP/dt
    # ---------------------------------------------------------

    # Control signal (PID output), limited by Max_reactivity
    control_signal = K_p*e + K_i*E
    rod_reactivity = np.minimum(alpha_h*control_signal, Max_reactivity)

    # Reactor reactivity term in dP/dt
    reactivity = ( rho_0
                   + rod_reactivity
                   + alpha_f*(T_f - T_f0)
                   + alpha_c*(T_c - T_c0)
                   - beta )

    # 1) dP/dt
    dP_dt = reactivity*(P/Lambda) + lambd*C

    # 2) dC/dt
    dC_dt = (beta*P/Lambda) - lambd*C

    # 3) dT_f/dt
    dTf_dt = (P/(M_f*C_f)) - (K*(T_f - T_c)/(M_f*C_f))

    # 4) dT_c/dt
    dTc_dt = (K*(T_f - T_c)/(M_c*C_c)) - (2*G*(T_c - T_in)/M_c)

    # 5) de/dt = ramp_slope - dP_dt
    de_dt = ramp_slope - dP_dt

    # 6) dE/dt = e
    dE_dt = e

    return [dP_dt, dC_dt, dTf_dt, dTc_dt, de_dt, dE_dt]


# ---------------------------------------------------------
# Choose parameter values
# ---------------------------------------------------------
rho_0       = 0.0000       # base reactivity
alpha_h     = 10e-5          # reactivity per
alpha_f     = -0.1929e-5        # feedback from fuel temperature
alpha_c     = -2e-5         # feedback from coolant temperature
beta        = 0.0310       # delayed neutron fraction
Lambda      = 0.8e-6       # neutron generation time
lambd       = 0.0001       # precursor decay constant
M_f         = 2132       # fuel mass (kg)
C_f         = 376       # fuel heat capacity (J/kg-K)
M_c         = 5429       # coolant mass (kg)
C_c         = 146       # coolant heat capacity (J/kg-K)
K           = 600000         # heat transfer coefficient (W/K)
G           = 25757          # coolant mass flow rate factor
T_in        = 400.0        # inlet coolant temperature (K)
T_f0        = 900.0        # reference T_f for reactivity feedback
T_c0        = 400.0        # reference T_c for reactivity feedback

# PID gains :
K_p         = 0.05
K_i         = 0.005

Max_reactivity = 0.001     # cap on total rod reactivity from controller
ramp_slope     = 1.0       # [MW/min] desired slope of power

# Pack parameters into a tuple for ODE solver
params = (rho_0, alpha_h, alpha_f, alpha_c, beta, Lambda, lambd,
          M_f, C_f, M_c, C_c, K, G, T_in, T_f0, T_c0,
          K_p, K_i, Max_reactivity, ramp_slope)

# ---------------------------------------------------------
# Initial conditions
#   y = [P,     C,    T_f,   T_c,   e,        E ]
# ---------------------------------------------------------
P_init   = 300.0   # [MW] example initial power
C_init   = beta*P_init/(Lambda*lambd)  # approximate initial precursor
T_f_init = 900.0
T_c_init = 400.0
e_init   = 0.0    # e(0) = P_set(0) - P(0) => if setpoint starts at P_init, e(0)=0
E_init   = 0.0

y0_initial = [P_init, C_init, T_f_init, T_c_init, e_init, E_init]

# Time span in minutes (e.g., 0 to 60 minutes)
t_span = (0, 60)
t_eval = np.linspace(t_span[0], t_span[1], 301)

# ---------------------------------------------------------
# Solve ODE
# ---------------------------------------------------------
sol = solve_ivp(
    fun=lambda t, y: reactor_odes(t, y, params),
    t_span=t_span,
    y0=y0_initial,
    t_eval=t_eval,
    rtol=1e-6, atol=1e-8
)

# Unpack solution
P_sol   = sol.y[0, :]
C_sol   = sol.y[1, :]
Tf_sol  = sol.y[2, :]
Tc_sol  = sol.y[3, :]
e_sol   = sol.y[4, :]
E_sol   = sol.y[5, :]
time    = sol.t

# ---------------------------------------------------------
# Compute the setpoint for plotting
# ---------------------------------------------------------
P0 = y0_initial[0]
P_set_sol = P0 + ramp_slope * time

# ---------------------------------------------------------
# Plot results
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))

# Plot power vs. time
plt.subplot(2,1,1)
plt.plot(time, P_sol, label='Power (MW)')
plt.plot(time, P_set_sol, 'r--', label='Setpoint (MW)')
plt.title('Reactor Power vs Time')
plt.xlabel('Time [min]')
plt.ylabel('Power [MW]')
plt.legend()
plt.grid(True)

# Plot control error
plt.subplot(2,1,2)
plt.plot(time, e_sol, label='Error = P_set - P')
plt.title('Control Error vs Time')
plt.xlabel('Time [min]')
plt.ylabel('Error [MW]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()