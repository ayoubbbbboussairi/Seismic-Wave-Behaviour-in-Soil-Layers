
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LeadFastReactor:
    def __init__(self):
        # Delayed neutron parameters
        self.beta = 319e-5  # Total delayed neutron fraction
        self.Lambda = 0.8066e-5  # Mean neutron generation time (s)

        # Delayed neutron group parameters
        self.beta_i = np.array([6.142e-5, 71.40e-5, 34.86e-5,
                                   114.1e-5, 69.92e-5, 22.68e-5])
        self.lambda_i = np.array([0.0125, 0.0292, 0.0895,
                                   0.2575, 0.6037, 2.6688])

        # Thermal parameters
        self.Mf = 2132.0  # Fuel mass (kg)
        self.Mc = 5429.0   # Coolant mass (kg)
        self.Cf = 376.0   # Fuel specific heat (J/kg/K)
        self.Cc = 146.0   # Coolant specific heat (J/kg/K)
        self.K = 600.0    # Heat transfer coefficient (W/K)
        self.G = 25757.0    # Coolant mass flow rate (kg/s)

        # Temperature feedback coefficients (pcm/K)
        self.alpha_D = -0.17e-5  # Doppler
        self.alpha_C = -1.995e-5  # Coolant density
        self.alpha_A = -0.2374e-5  # Axial expansion
        self.alpha_R = -0.7144e-5  # Radial expansion

        # Reference temperatures
        self.Tf0 = 900  # Reference fuel temperature (K)
        self.Tc0 = 400  # Reference coolant temperature (K)
        self.TcN = 417  # Nominal coolant temperature (K)

        # Time constants
        self.tau_f = self.Mf * self.Cf / self.K
        self.tau_c = self.Mc * self.Cc / self.K
        self.tau_0 = self.Mc / self.G

        # Nominal power
        self.P0 = 1500e6  # Nominal power (W)

    def reactivity(self, Tf, Tc, rho_ext):
        """Calculate total reactivity"""
        dTf = Tf - self.Tf0
        dTc = Tc - self.Tc0

        return (rho_ext +
                self.alpha_D * dTf +
                self.alpha_C * dTc +
                self.alpha_A * dTf +
                self.alpha_R * dTc)  # Convert pcm to absolute

    def derivatives(self, t, y, P0, rho_ext):
        """System of differential equations"""
        n = y[0]  # Neutron density
        c = y[1:7]  # Delayed neutron precursor concentrations
        Tf = y[7]  # Fuel temperature
        Tc = y[8]  # Average coolant temperature

        # Calculate reactivity
        rho = self.reactivity(Tf, Tc, rho_ext)

        # Calculate instantaneous power
        P = P0 * n

        # Neutronics equations
        dn_dt = (rho - self.beta) * n / self.Lambda + np.sum(self.lambda_i * c)
        dc_dt = self.beta_i * n / self.Lambda - self.lambda_i * c

        # Thermal equations
        dTf_dt = P / (self.Mf * self.Cf) - (Tf - Tc) / self.tau_f
        dTc_dt = (Tf - Tc) / self.tau_c - 2 * (Tc - self.TcN) / self.tau_0

        return np.concatenate(([dn_dt], dc_dt, [dTf_dt], [dTc_dt]))

    def simulate(self, t_span, P0, rho_ext, n0=1.0):
        """Run simulation for given time span and power level"""
        # Initial conditions
        c0 = self.beta_i * n0 / (self.Lambda * self.lambda_i)  # Equilibrium precursor concentrations
        Tf0 = self.Tf0
        Tc0 = self.Tc0
        y0 = np.concatenate(([n0], c0, [Tf0], [Tc0]))

        # Solve system
        sol = solve_ivp(
            fun=lambda t, y: self.derivatives(t, y, P0, rho_ext),
            t_span=t_span,
            y0=y0,
            method='LSODA',
            rtol=1e-6,
            atol=1e-8
        )

        return sol

    def plot_results(self, sol, P0):
        """Plot simulation results with improved scaling"""
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))

        # Calculate power
        power = P0 * sol.y[0]

        # Neutron density
        axs[0, 0].plot(sol.t, sol.y[0], 'b-', linewidth=2)
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Relative Neutron Density')
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Neutron Density Response')

        # Power
        axs[0, 1].plot(sol.t, power/1e6, 'r-', linewidth=2)  # Convert to MW
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Power (MW)')
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Reactor Power Response')

        # Precursor concentrations
        for i in range(6):
            axs[1, 0].plot(sol.t, sol.y[i+1], label=f'Group {i+1}', linewidth=2)
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Precursor Concentration')
        axs[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Delayed Neutron Precursor Response')

        # Fuel temperature
        axs[1, 1].plot(sol.t, sol.y[7], 'g-', linewidth=2)
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Temperature (K)')
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Fuel Temperature Response')

        # Coolant temperature
        axs[2, 0].plot(sol.t, sol.y[8], 'm-', linewidth=2)
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Temperature (K)')
        axs[2, 0].grid(True)
        axs[2, 0].set_title('Coolant Temperature Response')

        # Calculate and plot reactivity
        rho = np.array([self.reactivity(tf, tc, rho_ext)
                       for tf, tc in zip(sol.y[7], sol.y[8])])
        axs[2, 1].plot(sol.t, rho , 'k-', linewidth=2)  # Convert to pcm
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Reactivity (pcm)')
        axs[2, 1].grid(True)
        axs[2, 1].set_title('Total Reactivity Response')

        # Improve layout
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Create reactor instance
    reactor = LeadFastReactor()

    # Simulation parameters
    t_span = (0, 50)  # 100 seconds simulation
    P0 = 300e6  # Initial power (W)
    rho_ext = 10e-10  # External reactivity insertion (pcm)

    # Run simulation
    solution = reactor.simulate(t_span, P0, rho_ext)

    # Plot results
    reactor.plot_results(solution, P0)
    plt.show()