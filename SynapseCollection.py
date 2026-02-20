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
        return self.voltage

    def reset(self):
        """ Reset the state variables to the initial conditions for each neurone """
        self.voltage = np.zeros(self.n)


# NOTE: THIS CODE IS NOT MY OWN - IT IS TAKEN FROM https://soney.github.io/snn-from-scratch/chapters/13%20-%20Neuron%20Collections.html AND SLIGHTLY MODIFIED TO MAKE SURE MY IMPLEMENTATION WORKS AS EXPECTED
def test_neurone_synapse_link():
    import matplotlib.pyplot as plt
    from LIFCollection import LIFCollection
    def getDecoders(neurons, minJ=-1, maxJ=1, stepSize=0.01):
        def analyticalRate(v_th, tau_ref, tau_rc, I):
            if I <= v_th: return 0
            else:         return 1 / (tau_ref - tau_rc * np.log(1 - v_th/I))
        

        inputs = np.arange(minJ, maxJ, stepSize)

        gain_matrix = np.tile(neurons.gains[:, np.newaxis], (1, len(inputs)))
        bias_matrix = np.tile(np.expand_dims(neurons.biases, axis=1), (1, len(inputs)))
        encoders_matrix = np.tile(neurons.encoders, (1, len(inputs)))

        I = inputs * gain_matrix * encoders_matrix + bias_matrix

        tuningCurves = np.vectorize(analyticalRate)(neurons.v_th[-1], neurons.tau_ref, neurons.tau_rc, I)

        A = np.array(tuningCurves)

        value = np.expand_dims(inputs, axis=1)

        Gamma = A @ A.T + np.identity(len(neurons.output))
        GammaInv = np.linalg.inv(Gamma)
        Upsilon = A @ value

        Phi = GammaInv @ Upsilon

        return Phi

    t_step = 0.001
    n = 100
    neurons = LIFCollection(n=n, T_step=t_step)
    synapses = SynapseCollection(n=n, T_step=t_step)

    def step(inputs):
        neuron_output = neurons.step(inputs)
        synapse_output = synapses.step(neuron_output)
        return synapse_output

    def signal(t):
        return np.sin(t)

    T = 2*np.pi

    t = np.arange(0, T, t_step)
    inputs = signal(t)

    outputs = []
    for i in inputs:
        output = step(i)
        outputs.append(output)


    Phi = getDecoders(neurons)

    plt.figure()
    plt.plot(t, outputs @ Phi, linewidth=0.8)
    plt.plot(t, inputs, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Signal vs Time')
    plt.legend(['Decoded Signal', 'Input Signal'])
    plt.show()

    plt.title("Neuron Voltage vs Time")
    plt.plot(t, outputs)
    plt.show()

if __name__ == "__main__":
    test_neurone_synapse_link()