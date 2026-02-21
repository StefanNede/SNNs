# Adaptive (Analytical) LIF 
import math
import numpy as np

class ALIF:
    def __init__(self, tau_rc=0.2, tau_ref=0.002, v_init=0, v_th=1, tau_inh=0.05, inc_inh=1.0):
        self.tau_rc = tau_rc # rate of decay
        self.tau_ref = tau_ref # refractory period
        self.v = v_init # initial potential
        self.v_th = v_th # threshold potential

        # vvv ADAPTIVE SECTION vvv
        self.inh = 0 # how much to increase v_th by on spike
        self.inc_inh = inc_inh # strength of adaptaton -> how much to increase self.inh by on spike
        self.tau_inh = tau_inh # (inverse) rate of decay for inh

        self.output = 0
        self.refractory_time = 0
        
    # advance 1 time step and return output of neurone
    def step(self, I, T_step): 
        self.refractory_time -= T_step # subtract the amount of time that passed from our refractory time

        # can we accept input
        # generate absolute_time variable now as need it for later 
        if self.refractory_time < 0:
            # the refractory period might've ended within the time step 
            actual_time = min(abs(self.refractory_time), T_step)
        else:
            actual_time = 0

        leak_factor = actual_time/self.tau_rc
        self.v = I + (self.v - I) * math.exp(-leak_factor) 

        if self.v > self.v_th + self.inh: # Voltage is above the threshold + inhibition amount
            spike_time = actual_time + self.tau_rc * math.log((self.v - I) / (self.v_th - I)) 
            self.refractory_time = self.tau_ref + spike_time - actual_time

            self.output = 1 / T_step                         # Fire
            self.v = 0                                       # Reset potential
        else:
            self.output = 0          # Don't fire

        # decay inhibition amount, adding self.inc_inh only if neurone spiked
        self.inh = self.inh * math.exp(-T_step / self.tau_inh) + self.inc_inh * (self.output > 0)

        return self.output

    # reset neurone to initial state
    def reset(self):
        self.output = self.refractory_time = 0
        self.v = self.v_init
    
def visualise():
    import matplotlib.pyplot as plt

    T = 5
    t_step = 0.001
    neurone = ALIF(tau_rc=0.04, tau_ref=0.002, tau_inh=0.3, inc_inh=0.2)

    times = np.arange(0, T, t_step)
    inp = 1.3

    outputs = []
    for t in times:
        neurone.step(inp, t_step)
        outputs.append((neurone.v, neurone.v_th + neurone.inh, neurone.output))

    short_n = round(0.5 / t_step) # to see first part of simulation -> spike rate decreasing as neurone adapts to high input
    short_n = len(times) # to see full simulation -> spile rate becoming constant again

    plt.figure()
    plt.plot(times[:short_n], [o[0] for o in outputs[:short_n]], color='red', label='Voltage')
    plt.plot(times[:short_n], [o[1] for o in outputs[:short_n]], color='blue', label='Threshold', linestyle='--')
    plt.plot(times[:short_n], [o[2] * t_step for o in outputs[:short_n]], linewidth=4, color='black', label='Output')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualise()