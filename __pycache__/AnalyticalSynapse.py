import math

class AnalyticalSynapse:
    def __init__(self, tau_s=0.05):
        self.tau_s = tau_s
        self.g = 0 # g[0] = 0
    
    def step(self, I, T_step):
        self.output = I + math.exp(-T_step / self.tau_s) * (self.output - I)
        return self.output
    
    def reset(self):
        self.g = 0