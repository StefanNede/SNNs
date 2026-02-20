# Vectorised collection of analytical synapses
import numpy as np

# TODO: complete docstrings
class SynapseCollection:
    def __init__(self, n=1, tau_s=0.05, T_step=0.001):
        """
        Args:
            n (int, optional): _description_. Defaults to 1.
            tau_s (float, optional): _description_. Defaults to 0.05.
            T_step (float, optional): _description_. Defaults to 0.001.
        """
        self.n = n
        self.a = np.exp(-T_step / tau_s) # decay factor for synaptic current <- precalculated for performance gains
        self.b = 1 - self.a # scale factor for input current

        self.voltage = np.zeros(n) # initial voltage of neurons

    def step(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_
        """
        self.voltage = self.a * self.voltage + self.b * inputs

    def reset(self):
        """ Reset the state variables to the initial conditions for each neurone """
        self.voltage = np.zeros(n)