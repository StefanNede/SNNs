import numpy as np
import matplotlib.pyplot as plt

from FirstOrderLIF import FirstOrderLIF

class FirstOrderSynapse:
    def __init__(self, tau_s=0.01):
        self.tau_s = tau_s
        self.g = 0 # g[0] = 0
    
    def step(self, input_current, T_step):
        # discrete update equation here 
        self.g = self.g * (1 - T_step/self.tau_s) + input_current * T_step/self.tau_s
        return self.g
    
    def reset(self):
        self.g = 0

# neuron and synapse combined step 
def combinedStep(neuron, synapse, input_current, T_step):
    neuron_output = neuron.step(input_current, T_step)
    synapse_output = synapse.step(neuron_output, T_step)
    return synapse_output

if __name__ == "__main__":
    T_step = 0.001
    times = np.arange(0, 18, T_step)
    input_current = 2

    neuron = FirstOrderLIF(tau_rc=0.02, tau_ref=0.2)
    synapse = FirstOrderSynapse(tau_s=1.0)

    output_history = []

    for t in times:
        output = combinedStep(neuron, synapse, input_current, T_step)
        output_history.append(output)
    
    # plot
    plt.figure()
    plt.plot(times, output_history)
    plt.xlabel("Time")
    plt.ylabel("Synapse Output")
    plt.grid(True)
    plt.title("Synapse Output by Time")
    plt.show() 