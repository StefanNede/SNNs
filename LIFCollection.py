# Vectorised collection of Analytical LIFs 
import numpy as np

class LIFCollection:
    def __init__(self, n=1, dim=1, tau_rc=0.02, tau_ref=0.002, v_th=1, max_rates=[200,400], intercept_range=[-1,1], T_step=0.001, v_init=0):
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
        """
        self.n = n

        # Neurone parameters ---
        self.dim = dim
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.v_th = np.ones(n) * v_th
        self.t_step = T_step
        self.v_init = v_init
        
        # State variables ---
        self.voltage = np.ones(n) * v_init
        self.refractory_time = np.zeros(n) # time remaining in refractory period
        self.output = np.zeros(n)  # output spikes

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
        # There is an error in the next 2 lines, so is replaced with correct code on line 67:
        dot_product = np.sum(inputs * self.encoders, axis=1) # do dot product of each input along each encoder (axis=1 tells it to go row-wise in self.encoders)
        I = dot_product * self.gains + self.biases
        I = np.sum(self.biases + inputs * self.encoders * self.gains[:, np.newaxis], axis=0) / self.n

        # Update membrane potential
        leak_factor = delta_t/self.tau_rc
        self.voltage = I + (self.voltage - I) * np.exp(-leak_factor)

        # Determine which neurones spike
        spike_mask = self.voltage > self.v_th
        self.output[:] = spike_mask / self.t_step # using in-place update here for performance <- using discrete approximation of dirac delta function

        # Calculate time spike occurs in time step
        spike_time = self.tau_rc * np.log((self.voltage[spike_mask] - I[spike_mask]) / (self.v_th[spike_mask] - I[spike_mask])) + delta_t[spike_mask]
        self.refractory_time[spike_mask] = self.tau_ref + spike_time - delta_t[spike_mask] # set refractory time countdown for spiking neurones

        # Reset voltage of spiking neurones
        self.voltage[spike_mask] = 0

        return self.output

    def reset(self):
        """ Reset the state variables to the initial conditions for each neurone """
        self.voltage = np.ones(self.n) * self.v_init
        self.refractory_time = np.zeros(self.n)
        self.output = np.zeros(self.n)
    
def test_responses():
    import matplotlib.pyplot as plt

    T_step = 0.001
    neurones = LIFCollection(n=10, dim=1, T_step=T_step)

    def compute_response(neurone, inputs, T = 10):
        spike_count = np.zeros(len(neurone.output))
        for _ in np.arange(0, T, T_step):
            output = neurone.step(inputs)
            spike_count += output * T_step
        return spike_count / T

    inputs = np.arange(-1, 1, 0.05)
    responses = []
    for i in inputs:
        response = compute_response(neurones, i)
        responses.append(response)

    plt.plot(inputs, responses)
    plt.show()

if __name__ == "__main__":
    test_responses()