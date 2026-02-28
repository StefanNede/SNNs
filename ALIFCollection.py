# Vectorised Collection of Adaptive (Analytical) LIF 
import numpy as np

class ALIFCollection:
    def __init__(self, n=1, dim=1, tau_rc=0.02, tau_ref=0.002, v_th=1, max_rates=[200,400], intercept_range=[-1,1], T_step=0.001, v_init=0, tau_inh=0.05, inc_inh=1.0):
        """
        Args:
            n (int, optional): number of neurones in collection
            dim (int, optional): dimensionality of the input
            tau_rc (float, optional): membrane time constant
            tau_ref (float, optional): refractory period
            v_th (int, optional): threshold voltage for spiking
            max_rates (list, optional): max rate for neurones firing rate to output (behavioural property used to calculate gains and biases)
            intercept_range (list, optional): when the neurone starts firing based on modified input (behavioural property used to calculate gains and biases)
            T_step (float, optional): time step for simulation
            v_init (int, optional): initial voltage of neurones
            vvv ADDED TO MAKE NEURONES ADAPTIVE vvv
            tau_inh (float, optional): (inverse) rate of decay for inh (higher => inhibition remains for longer)
            inc_inh(float, optional): strength of adaptaton -> how much to increase self.inh by on spike (higher => inhibition stronger)
        """
        self.n = n

        # Neurone parameters ---
        self.dim = dim
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.tau_inh = tau_inh
        self.v_th = np.ones(n) * v_th
        self.t_step = T_step
        self.inc_inh = inc_inh
        self.v_init = v_init
        
        # State variables ---
        self.voltage = np.ones(n) * v_init
        self.refractory_time = np.zeros(n) # time remaining in refractory period
        self.output = np.zeros(n)  # output spikes
        self.inh = np.zeros(n) # inhibition

        # Generate random max rates and intercepts within the given range
        max_rates_vector = np.random.uniform(max_rates[0], max_rates[1], n)
        intercepts_vector = np.random.uniform(intercept_range[0], intercept_range[1], n)

        # Calculate gain and bias for each neurone <- vectorised version of getGainBias() function
        self.gains = self.v_th * (1 - 1 / (1 - np.exp((self.tau_ref - 1/max_rates_vector) / self.tau_rc))) / (intercepts_vector - 1)
        self.biases = self.v_th - self.gains * intercepts_vector
        
        # Initialise random encoders = direction of each input neurone reacts to
        self.encoders = np.random.randn(n, self.dim)
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, np.newaxis] # standardise each row (encoder_i) independently

    def step(self, inputs):
        """ Advance 1 time step (self.t_step) analytically, and return output of each neurone integrating new input (potentially firing or not)

        Args:
            inputs (Array[Float] | Float): input flowing into neurones (to be interpreted by each one in collection differently) -> has dimension self.dim
        """
        self.refractory_time -= self.t_step

        # Analogous to actual_time in AnalyticalLIF
        delta_t = np.where(
            self.refractory_time < 0, 
            np.minimum(np.abs(self.refractory_time), self.t_step), # refractory period ended (maybe during time step)
            0.0 # refractory period not ended -> so don't want to integrate any time
        )

        # Calculate input current (by applying encoders, gains and biases)
        #    - dot product of each input with each encoder -> array of e_i . input_i
        #    - then element-wise multiplication by self.gains_i 
        #    - and element-wise addition by self.biases_i
        # There is an error in the next 2 lines, so is replaced with correct code on line 73:
        # dot_product = np.sum(inputs * self.encoders, axis=1) # do dot product of each input along each encoder (axis=1 tells it to go row-wise in self.encoders)
        # I = dot_product * self.gains + self.biases
        I = np.sum(self.biases + inputs * self.encoders * self.gains[:, np.newaxis], axis=0) / self.n

        # Update membrane potential
        leak_factor = delta_t/self.tau_rc
        self.voltage = I + (self.voltage - I) * np.exp(-leak_factor)

        # Determine which neurones spike
        spike_mask = self.voltage > self.v_th + self.inh # when greater than threshold + inhibition amount
        self.output[:] = spike_mask / self.t_step # using in-place update here for performance <- using discrete approximation of dirac delta function

        # Calculate time spike occurs in time step
        spike_time = self.tau_rc * np.log((self.voltage[spike_mask] - I[spike_mask]) / (self.v_th[spike_mask] - I[spike_mask])) + delta_t[spike_mask]
        self.refractory_time[spike_mask] = self.tau_ref + spike_time - delta_t[spike_mask] # set refractory time countdown for spiking neurones

        # Reset voltage of spiking neurones
        self.voltage[spike_mask] = 0

        # Update inhibition amount (adaptive part of ALIF)
        #   - decay inhibition amount, adding self.inc_inh only if neurone spiked
        self.inh = self.inh * np.exp(-self.t_step/self.tau_inh) + self.inc_inh * (self.output > 0)

        return self.output

    def reset(self):
        """ Reset the state variables to the initial conditions for each neurone """
        self.voltage = np.ones(self.n) * self.v_init
        self.refractory_time = np.zeros(self.n)
        self.output = np.zeros(self.n)
        self.inh = np.zeros(self.n)

# Showing how ALIFs more resistant to input variability compared to LIFs
def variability_resistance_test():
    from LIFCollection import LIFCollection
    import matplotlib.pyplot as plt
    def plotFiringRates(tau_inh, inc_inh, title=""):
        t_step = 0.001
        def getFiringRate(inp, neuron, runtime=10):
            times = np.arange(0, runtime, t_step)
            num_spikes = 0
            for t in times:
                neuron.step(inp)
                if neuron.output[0] > 0:
                    num_spikes += 1
            return num_spikes / runtime

        alifs = ALIFCollection(n=1, tau_ref=0.002, tau_rc=0.2, T_step=t_step, tau_inh=tau_inh, inc_inh=inc_inh)
        lifs  =  LIFCollection(n=1, tau_ref=0.002, tau_rc=0.2, T_step=t_step)

        alifs.biases = lifs.biases = np.array([0]) ; alifs.gains = lifs.gains = np.array([1]) ; alifs.encoders = lifs.encoders = np.array([[1]]) # Remove the random gain, bias, and encoders

        rates = []

        for inp in np.linspace(0.9, 2, 100):
            alifFiringRate = getFiringRate(inp, alifs)
            lifFiringRate  = getFiringRate(inp, lifs)

            alifs.reset() ; lifs.reset()
            rates.append((inp, alifFiringRate, lifFiringRate))

        plt.figure()
        plt.plot([r[0] for r in rates], [r[1] for r in rates], label='ALIF')
        plt.plot([r[0] for r in rates], [r[2] for r in rates], label='LIF')
        plt.title(title)
        plt.xlabel("Input Magnitude")
        plt.ylabel("Spiking Rate")
        plt.legend()
        plt.show()
    
    # high tau_inh => inhibition lasts longer
    # high inc_inh => stronger inhibition

    plotFiringRates(tau_inh=0.7, inc_inh=1.0, title="High + Long Lasting Inhibition")
    plotFiringRates(tau_inh=0.3, inc_inh=1.0, title="High + Short Lasting Inhibition")
    plotFiringRates(tau_inh=0.3, inc_inh=0.1, title="Low + Short Lasting Inhibition")

if __name__ == "__main__":
    variability_resistance_test()