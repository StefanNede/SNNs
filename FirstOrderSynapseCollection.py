import numpy as np
import matplotlib.pyplot as plt
import math

from FirstOrderLIFCollection import FirstOrderLIFCollection
from FirstOrderSynapse import FirstOrderSynapse
from TuningCurves import computeCollectionDecoder

class FirstOrderSynapseCollection:
    def __init__(self, num_neurones, tau_s=0.01):
        self.synapses = [FirstOrderSynapse(tau_s) for _ in range(num_neurones)]
    
    def step(self, input_currents, T_step):
        outputs = []
        for i, synapse in enumerate(self.synapses):
            inp = input_currents[i]
            output = synapse.step(inp, T_step)
            outputs.append(output)
        
        return outputs

    def reset(self):
        for synapse in self.synapses:
            synapse.reset()

def combinedStepCollection(neurones, synapses, input_currents, T_step):
    neurone_outputs = neurones.step(input_currents, T_step)
    synapse_outputs = synapses.step(neurone_outputs, T_step)
    return synapse_outputs

# make a function that takes neurones and outputs of neurone-synapses to regenerate signal 
def getSignal():
    pass

if __name__ == "__main__":
    print("TESTING ON SIN WAVE AS INPUT SIGNAL")

    def signal(t):
        return math.sin(t)

    NUM_NEURONES = 45
    neurones = FirstOrderLIFCollection(NUM_NEURONES, max_rate_range=(25, 100))
    synapses = FirstOrderSynapseCollection(NUM_NEURONES, tau_s=0.05)

    T_step = 0.001
    T = np.arange(0, 12, T_step)

    signals = []
    outputs = []
    for t in T:
        input_value = signal(t)
        out = combinedStepCollection(neurones, synapses, input_value, T_step)
        signals.append(input_value)
        outputs.append(out)

    plt.figure()
    plt.plot(T, signals, color="C1")
    plt.xlabel('Time (s)')
    plt.ylabel('Input Signal')
    plt.show()

    plt.figure()
    plt.plot(T, outputs)
    plt.xlabel('Time (s)')
    plt.ylabel('Outputs')
    plt.show()
    
    Phi = computeCollectionDecoder(neurones)
    print(f"Decoders are: {Phi.flatten()}")

    # vector multiply outputs by decoders to get orignal signal
    plt.figure()
    plt.plot(T, (outputs @ Phi).flatten()) # Multiply the outputs by the decoders with matrix multiplication
    plt.plot(T, signals, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Signal vs Time')
    plt.legend(['Decoded Signal', 'Input Signal'])
    plt.show()