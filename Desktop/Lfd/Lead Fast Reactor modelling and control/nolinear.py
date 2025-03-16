
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lfr_nonlinear_odes(t, y, params):
    """
    Nonlinear ODE system for a lumped-parameter LFR model (6 delayed neutron groups).

    States:
      y[0] = psi (neutron population or normalized power)
      y[1..6] = eta1..eta6 (delayed neutron precursors)
      y[7] = T_f (fuel temperature)
      y[8] = T_c (coolant temperature)
    """
  
    psi = y[0]
    eta = y[1:7]
    T_f = y[7]
    T_c = y[8]

    # Unpack parameters
    beta       = params['beta']           # total delayed fraction
    beta_i     = params['beta_i']         # array of length 6
    lam        = params['lam']            # array of length 6
    Lambda     = params['Lambda']         # prompt neutron generation time
    alpha_h    = params['alpha_h']        # rod worth coefficient
    alpha_f    = params['alpha_f_total']  # combined Doppler + axial
    alpha_c    = params['alpha_c_total']  # combined coolant + radial
    T_f0       = params['T_f0']
    T_c0       = params['T_c0']
    h          = params['h']             # control rod movement
    tau_f      = params['tau_f']
    tau_c      = params['tau_c']
    tau_0      = params['tau_0']
    T_in       = params['T_in']          # inlet temperature
    P0_K_tau_f = params['P0_K_tau_f']    # the constant power term

    # Reactivity feedback
    rho = alpha_h*h + alpha_f*(T_f - T_f0) + alpha_c*(T_c - T_c0)

    # 1) Neutron population
    # dpsi/dt = [(rho - beta)/Lambda]*psi + sum_i [beta_i / Lambda * eta_i]
    dpsi = ((rho - beta) / Lambda)*psi + np.sum((beta_i / Lambda)*eta)

    # 2) Delayed neutron precursors
    # deta_i/dt = lam_i * psi - lam_i * eta_i
    deta = lam*psi - lam*eta

    # 3) Fuel temperature ODE :
    # dT_f/dt = P0_K_tau_f - (1/tau_f)*T_f + (1/tau_f)*T_c
    dTf = P0_K_tau_f - (1.0 / tau_f)*T_f + (1.0 / tau_f)*T_c

    # 4) Coolant temperature
    # dT_c/dt = (1/tau_c)*(T_f - T_c) - (2/tau_0)*(T_c - T_in)
    dTc = (1.0 / tau_c)*(T_f - T_c) - (2.0 / tau_0)*(T_c - T_in)

    return np.concatenate(([dpsi], deta, [dTf, dTc]))


def simulate_scenario(params, t_span=(0, 30), num_points=600):
    """
    Solves the nonlinear LFR ODE system for given parameters.
    Returns time vector and state solutions plus T_out, rho.
    """
    # Initial conditions (assuming near steady-state)
    psi0 = 1.0
    eta0 = (params['beta_i'] / params['beta']) * psi0
    y0 = np.zeros(9)
    y0[0] = psi0
    y0[1:7] = eta0
    y0[7] = params['T_f0']
    y0[8] = params['T_c0']

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(
        fun=lambda t, y: lfr_nonlinear_odes(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )

    # Extract solution
    t = sol.t
    psi_sol = sol.y[0, :]
    eta_sol = sol.y[1:7, :]
    T_f_sol = sol.y[7, :]
    T_c_sol = sol.y[8, :]

    # Compute reactivity
    alpha_h = params['alpha_h']
    alpha_f = params['alpha_f_total']
    alpha_c = params['alpha_c_total']
    h       = params['h']
    T_f0    = params['T_f0']
    T_c0    = params['T_c0']
    rho_sol = alpha_h*h + alpha_f*(T_f_sol - T_f0) + alpha_c*(T_c_sol - T_c0)

    # Compute coolant outlet temperature
    T_out_sol = 2.0*T_c_sol - params['T_in']

    return t, psi_sol, eta_sol, T_f_sol, T_c_sol, T_out_sol, rho_sol


def plot_results(t, psi, eta, T_f, T_c, T_out, rho, scenario_title="Scenario"):
    """
    Plot results for a single scenario.
    """
    fig, axs = plt.subplots(2, 2, figsize=(11,8))
    fig.suptitle(scenario_title, fontsize=14)

    # (1) Neutron population
    axs[0,0].plot(t, psi, 'r', lw=2)
    axs[0,0].set_xlabel('Time [s]')
    axs[0,0].set_ylabel(r'$\psi$ (power)')
    axs[0,0].grid(True)

    # (2) Delayed neutron precursors
    for i in range(6):
        axs[0,1].plot(t, eta[i, :], label=f'$\eta_{i+1}$')
    axs[0,1].set_xlabel('Time [s]')
    axs[0,1].set_ylabel('Precursors')
    axs[0,1].grid(True)
    axs[0,1].legend(loc='best')

    # (3) Fuel, Coolant, Outlet temperatures
    axs[1,0].plot(t, T_f, 'b', label='Fuel temp')
    axs[1,0].plot(t, T_c, 'g', label='Coolant temp')
    axs[1,0].plot(t, T_out, 'm--', label='Coolant outlet temp')
    axs[1,0].set_xlabel('Time [s]')
    axs[1,0].set_ylabel('Temperature [°C]')
    axs[1,0].grid(True)
    axs[1,0].legend(loc='best')

    # (4) Reactivity
    axs[1,1].plot(t, rho, 'k', lw=2)
    axs[1,1].set_xlabel('Time [s]')
    axs[1,1].set_ylabel(r'Reactivity $\rho$ (dimensionless)')
    axs[1,1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # Common parameters
    params = {
        'beta':    319e-5,  # total delayed fraction
        'beta_i':  np.array([6.142e-5, 71.4e-5, 34.86e-5,
                             114.1e-5, 69.92e-5, 22.68e-5]),
        'lam':     np.array([0.0125, 0.0292, 0.0895, 0.2575, 0.6037, 2.6688]),
        'Lambda':  0.8066e-6,  # s
        # Feedback coefficients in dimensionless reactivity (1 pcm = 1e-5)
        'alpha_h':       10e-5,  #
        'alpha_f_total': (-0.15 - 0.0429)*1e-5,  #  (Doppler+axial)
        'alpha_c_total': (-1.2267 - 0.7741)*1e-5,#  (coolant+radial)
        'T_f0':  900.0,
        'T_c0':  400.0,
        'tau_f': 1.336,
        'tau_c': 1.321,
        'tau_0': 0.210,
        'T_in':  400.0,
        # The corrected constant power term for the fuel ODE:
        # dT_f/dt = P0_K_tau_f - (1/tau_f)*T_f + (1/tau_f)*T_c
        'P0_K_tau_f': 374.25,  # you can tune this value
        # We'll define rod movement h=0 by default
        'h': 0.0
    }

     # ------------------------------------------------------------------
    # Scenario 1:  dh = 1 pcm => rod insertion of +1 => reactivity +1 pcm
    #  alpha_h=1e-5 => multiply by h=1 => total rod reactivity = 1e-5 (1 pcm)
    #  keep T_in = 400
    # ------------------------------------------------------------------
    params_s1 = dict(params)  # copy
    params_s1['h'] = 10.0      # +1 => +1 pcm
    params_s1['T_in'] = 400.0 # no change in inlet T

    t1, psi1, eta1, Tf1, Tc1, Tout1, rho1 = simulate_scenario(params_s1)
    plot_results(t1, psi1, eta1, Tf1, Tc1, Tout1, rho1,
                 scenario_title="Scenario 1: +1 pcm rod insertion")

    # ------------------------------------------------------------------
    # Scenario 2:  dh = 100 pcm => rod insertion of +100 => reactivity +100 pcm
    #  alpha_h=1e-5 => multiply by h=100 => total rod reactivity= 1e-3 => 100 pcm
    #  keep T_in = 400
    # ------------------------------------------------------------------
    params_s2 = dict(params)
    params_s2['h'] = 100.0   # +100 => +100 pcm
    params_s2['T_in'] = 400.0
    t2, psi2, eta2, Tf2, Tc2, Tout2, rho2 = simulate_scenario(params_s2)
    plot_results(t2, psi2, eta2, Tf2, Tc2, Tout2, rho2,
                 scenario_title="Scenario 2: +100 pcm rod insertion")

    # ------------------------------------------------------------------
    # Scenario 3:  dTin = +20°C => T_in=420°C, no rod insertion
    # ------------------------------------------------------------------
    params_s3 = dict(params)
    params_s3['h'] = 0.0        # no rod movement
    params_s3['T_in'] = 420.0   # +20 from 400
    t3, psi3, eta3, Tf3, Tc3, Tout3, rho3 = simulate_scenario(params_s3)
    plot_results(t3, psi3, eta3, Tf3, Tc3, Tout3, rho3,
                 scenario_title="Scenario 3: +20°C in coolant inlet")

if __name__ == "__main__":
    main()