
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def reactor_odes(t, y, params):

    P   = y[0]
    c1  = y[1]
    c2  = y[2]
    c3  = y[3]
    c4  = y[4]
    c5  = y[5]
    c6  = y[6]
    Tf  = y[7]
    Tc  = y[8]

    # Unpack parameters
    alpha_f = params['alpha_f']   # fuel feedback coefficient
    alpha_c = params['alpha_c']   # coolant feedback coefficient
    alpha_ext = params['alpha_ext']  # external reactivity
    beta  = params['beta']        # total delayed fraction
    bi    = params['bi']          # array of delayed fractions per group
    lambd = params['lambd']       # array of decay constants
    L     = params['L']           # neutron generation time
    # Reference T for feedback
    Tf0   = params['Tf0']
    Tc0   = params['Tc0']

    #   rho_total = alpha_ext + alpha_f*(Tf - Tf0) + alpha_c*(Tc - Tc0)
    rho_total = alpha_ext + alpha_f*(Tf - Tf0) + alpha_c*(Tc - Tc0)

    #   dP/dt = [(rho_total - beta)/L]*P + sum(lambd[i]*c_i)
    dc_sum = lambd[0]*c1 + lambd[1]*c2 + lambd[2]*c3 + lambd[3]*c4 + lambd[4]*c5 + lambd[5]*c6
    dPdt   = ((rho_total - beta)/L)*P + dc_sum

    # Delayed precursors:
    #   dci/dt = (bi[i]/L)*P - lambd[i]*ci
    dc1dt = (bi[0]/L)*P - lambd[0]*c1
    dc2dt = (bi[1]/L)*P - lambd[1]*c2
    dc3dt = (bi[2]/L)*P - lambd[2]*c3
    dc4dt = (bi[3]/L)*P - lambd[3]*c4
    dc5dt = (bi[4]/L)*P - lambd[4]*c5
    dc6dt = (bi[5]/L)*P - lambd[5]*c6

    # Simple thermal-hydraulics :
    #   dTf/dt = a*P - b*(Tf - Tc)
    #   dTc/dt = c*(Tf - Tc) - d*(Tc - Tc_in)
    # Adjust as needed for your real model.
    a = params['a']  #
    b = params['b']  #
    c = params['c']  # ...
    d = params['d']  # ...
    Tc_in = params['Tc_in']

    dTfdt = a*P - b*(Tf - Tc)
    dTcdt = c*(Tf - Tc) - d*(Tc - Tc_in)

    return [dPdt, dc1dt, dc2dt, dc3dt, dc4dt, dc5dt, dc6dt, dTfdt, dTcdt]

def run_simulation(alpha_c, alpha_f):
    """
    Runs the reactor ODE model for the given alpha_c, alpha_f.
    Returns (t, P(t)) so we can plot power vs time.
    """

    # --------------------------------------------------------------------
    # 1) Define parameters for the model 
    # --------------------------------------------------------------------
    params = {
        # feedback coefficients
        'alpha_c': alpha_c,
        'alpha_f': alpha_f,
        'alpha_ext': 0.0,      # external reactivity if any
        # delayed neutron data
        'beta': 319,        # total delayed fraction
        'bi': np.array([6.142, 71.40, 34.86,114.1, 69.92, 22.68]),  # 6-group
        'lambd': np.array([0.0125, 0.0292, 0.0895, 0.2575, 0.6037, 2.6688]),  # decay constants
        'L': 0.8e-6,           # neutron generation time
        # reference temps
        'Tf0': 900.0,
        'Tc0': 400.0,
        # thermal-hydraulic placeholders
        'a': 0.001247,   # coefficient for dTf/dt
        'b': 0.74,   # ...
        'c': 0.75,   # ...
        'd': 9.52,   # ...
        'Tc_in': 400.0  # inlet coolant temperature
    }

    # --------------------------------------------------------------------
    # 2) Define initial conditions
    #    Suppose we have 9 states: P, c1..c6, Tf, Tc
    # --------------------------------------------------------------------
    P0   = 3.0e8       # initial power
    c0   = np.zeros(6) # all precursors start at 0
    Tf0  = 900.0
    Tc0  = 400.0
    y0   = np.concatenate(([P0], c0, [Tf0, Tc0]))  # length=9

    # --------------------------------------------------------------------
    # 3) Solve ODE
    # --------------------------------------------------------------------
    t_span = (0, 50)                 # simulate 0..50 seconds
    t_eval = np.linspace(0, 50, 501) # 501 time points
    sol = solve_ivp(
        fun=lambda t, y: reactor_odes(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )

    # The first state (index=0) is Power => sol.y[0,:]
    return sol.t, sol.y[0,:]

def main():
    # We define 4 scenarios for (alpha_c, alpha_f):
    #   1) alpha_c=-1,   alpha_f=-1
    #   2) alpha_c= 1,   alpha_f=-1.5
    #   3) alpha_c= 0,   alpha_f= 1
    #   4) alpha_c= 2,   alpha_f= 2
    scenarios = [
        {'alpha_c': -1.0,  'alpha_f': -1.0,  'label': '(ac=-1, af=-1)'},
        {'alpha_c':  1.0,  'alpha_f': -1.5,  'label': '(ac=1, af=-1.5)'},
        {'alpha_c':  0.0,  'alpha_f':  1.0,  'label': '(ac=0, af=1)'},
        {'alpha_c':  2.0,  'alpha_f':  2.0,  'label': '(ac=2, af=2)'}
    ]

    plt.figure(figsize=(8,6))

    # Run each scenario and plot the power
    for scenario in scenarios:
        alpha_c = scenario['alpha_c']
        alpha_f = scenario['alpha_f']
        label   = scenario['label']

        t_sol, P_sol = run_simulation(alpha_c, alpha_f)

        P_sol_MW = P_sol / 1e6

        plt.plot(t_sol, P_sol_MW, label=label)

    plt.title("Reactor Power vs. Time (varying alpha_c, alpha_f)")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [MW]")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()