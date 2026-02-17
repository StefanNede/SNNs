import numpy as np
import matplotlib.pyplot as plt

from FirstOrderLIF import FirstOrderLIF, getGainBias
from TuningCurves import getAnalyticalFiringRate

class FirstOrderLIFCollection: # Collection of First Order Leaky Integrate and Fire Neurons
    def __init__(self, num_neurones, tau_rc=0.02, tau_ref=0.002, v_th=1, max_rate_range = (200, 400), intercept_range = (-1, 1), encoder_options = (-1, 1)):
        self.neurones  = [] # List of neurones
        self.gains    = [] # List of gains (numbers)
        self.biases   = [] # List of biases (numbers)
        self.encoders = [] # List of encoders (numbers)

        for _ in range(num_neurones):
            neurone = FirstOrderLIF(tau_rc=tau_rc, tau_ref=tau_ref, v_th=v_th)
            max_rate  = np.random.uniform(max_rate_range[0], max_rate_range[1])   # Maximum firing rate
            intercept = np.random.uniform(intercept_range[0], intercept_range[1]) # Intercept (where the neuron starts firing)
            encoder   = np.random.choice(encoder_options)                         # Encoder (direction of the neuron's tuning curve) 

            gain, bias = getGainBias(max_rate, intercept, tau_rc, tau_ref, v_th)

            self.neurones.append(neurone)
            self.gains.append(gain)
            self.biases.append(bias)
            self.encoders.append(encoder)
    
    def step(self, I, T_step):
        outputs = []
        for neurone, gain, bias, encoder in zip(self.neurones, self.gains, self.biases, self.encoders): # Loop over neurons, gains, biases, and encoders
            output = neurone.step(I * gain * encoder + bias, T_step) # Step the neuron with the input scaled by the gain and encoder, plus the bias
            outputs.append(output) # Append the output to the list of outputs
        return outputs
    
    def reset(self): # Reset all neurons to their initial state
        for neuron in self.neurones:
            neuron.reset()


def computeTuningCurves(lifs, inputs, time_limit = 1, t_step = 0.001):
    tuning_curves = [ [0] * len(inputs) for _ in range(len(lifs.neurones)) ]
    num_neurones = len(lifs.neurones)
    for inp_idx, inp in enumerate(inputs):
        spike_counts = [0] * num_neurones
        lifs.reset()
        for _ in np.arange(0, time_limit, t_step):
            outputs = lifs.step(inp, t_step)
            for i in range(num_neurones):
                if outputs[i] > 0:
                    spike_counts[i] += 1
        for i in range(num_neurones):
            tuning_curves[i][inp_idx] = spike_counts[i] / time_limit
    lifs.reset()

    return tuning_curves

if __name__ == "__main__":
    NUM_NEURONES = 15 
    lifs = FirstOrderLIFCollection(NUM_NEURONES, max_rate_range=(25, 100))

    inputs = np.arange(-1, 1, 0.01)

    plt.figure()
    # for curve in computeTuningCurves(lifs, inputs):
    #     plt.plot(inputs, curve)
    for neurone, gain, bias, encoder in zip(lifs.neurones, lifs.gains, lifs.biases, lifs.encoders):
        tuning_curve = [getAnalyticalFiringRate(neurone, i * gain * encoder + bias) for i in inputs]
        plt.plot(inputs, tuning_curve)

    plt.xlabel('Input Current')
    plt.ylabel('Firing Rate')
    plt.title('Tuning Curves')
    plt.show()