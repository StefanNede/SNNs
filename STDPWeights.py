# Spike-Timing Dependent Plasticity Weight Learning between neurone layers
import numpy as np

class STDPWeights:
    def __init__(self, numPre, numPost, tau_plus = 0.02, tau_minus = 0.02, a_plus = 0.01, a_minus = 0.011, g_min=0, g_max=1):
        """
        Args:
            numPre (int): number of pre-synaptic neurones
            numPost (int): number of post-synaptic neurones
            tau_plus (float, optional): time constant for the pre-synaptic trace -> controls decay speed of the strengthening weight update (higher => synapse can still be strengthened even if post spike occurs much later than pre spike)
            tau_minus (float, optional): time constant for the post-synaptic trace -> controls decay speed of the weakening weight update (higher => synapse can still be weakened even if pre spike occurs much later than post spike)
            a_plus (float, optional): learning rate for strengthening weights
            a_minus (float, optional): learning rate for weakening weights
            g_min (int, optional): lower bound for weight value / synaptic conductance strength
            g_max (int, optional): upper bound for weight value / synaptic conductance strength
        """
        self.numPre = numPre
        self.numPost = numPost
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.x = np.zeros(numPre)
        self.y = np.zeros(numPost)

        self.g_min = g_min
        self.g_max = g_max
        self.w = np.random.uniform(g_min, g_max, (numPre, numPost)) # 2D array; numPre x numPost zeros (random values between g_min and g_max)
    
    def step(self, t_step):
        """Move one time step forward -> apply exponential decay to the traces for each neurone 

        Args:
            t_step (float): time step of the simulation
        """
        self.x = self.x * np.exp(-t_step/self.tau_plus)
        self.y = self.y * np.exp(-t_step/self.tau_minus)
    
    def updateWeights(self, preOutputs, postOutputs):
        """Update the weights between the pre-synaptic and post-synaptic layer depending on STDP update and what neurones have fired (linear time in spike pairs)

        Args:
            preOutputs (Array[float]): the outputs of the pre-synaptic neurones
            postOutputs (Array[float]): the outputs of the post-synaptic neurones
        """
        # start new accumulating dw for each neurone that spiked (need to accumulate time from here onwards for future updates)
        self.x += (preOutputs  > 0) * self.a_plus 
        self.y -= (postOutputs > 0) * self.a_minus

        alpha_g = self.g_max - self.g_min # scaling factor for weight updates

        # get indices of pre and post synaptic spiking neurones
        preSpikeIndices = np.where(preOutputs > 0)[0]
        postSpikeIndices = np.where(postOutputs > 0)[0]

        # weight weakening updates 
        for ps_idx in preSpikeIndices:
            # use self.w[ps_idx] to update all outgoing connections of ps_idx pre-synaptic neurone (update row ps_idx of weights matrix)
            self.w[ps_idx] += alpha_g * self.y
            self.w[ps_idx] = np.clip(self.w[ps_idx], self.g_min, self.g_max) # make sure in conductance range

        # weight strengthening updates
        for ps_idx in postSpikeIndices:
            # use self.w[:, ps_idx] to update all incoming connections to ps_idx post-synaptic neurone (update column ps_idx of weights matrix)
            self.w[:, ps_idx] += alpha_g * self.x
            self.w[:, ps_idx] = np.clip(self.w[:, ps_idx], self.g_min, self.g_max) # make sure in conductance range