import numpy as np

# First-Order LIF with greater accuracy wrt refractory periods occurring in between time steps
class FirstOrderFractionalLIF:
    def __init__(self, tau_rc=0.2, tau_ref=0.002, v_init=0, v_th=1):
        self.tau_rc = tau_rc # rate of decay
        self.tau_ref = tau_ref # refractory period
        self.v = v_init # initial potential
        self.v_th = v_th # threshold potential
        self.v_init = v_init

        self.output = 0
        self.refractory_time = 0

    # advance 1 time step and return output of neurone
    def step(self, I, T_step): 
        self.refractory_time -= T_step

        # can we accept input
        if self.refractory_time < 0:
            # the refractory period might've ended within the time step 
            actual_time = min(abs(self.refractory_time), T_step)
            leak_factor = actual_time/self.tau_rc
            self.v = self.v*(1-leak_factor) + I*leak_factor # integrate input potential

        # print(self.v, self.v_th)
        # fire if potential above threshold else output = 0
        if self.v >= self.v_th: 
            # better refractory_time for if we cross the threshold potential
            # during time step t -> t + t_step (must account that some refractory_time already passed)
            spike_time = (self.tau_rc * (self.v_th - self.v) + T_step * (I - self.v_th)) / (I - self.v)
            self.refractory_time = self.tau_ref + spike_time - T_step
            self.output = 1/T_step
            self.v = 0
        else:
            self.output = 0
        
        return self.output
    
    # reset neurone to initial state
    def reset(self):
        self.output = self.refractory_time = 0
        self.v = self.v_init