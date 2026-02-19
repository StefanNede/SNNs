import math
import matplotlib.pyplot as plt
import numpy as np
from FirstOrderLIF import FirstOrderLIF
from FirstOrderFractionalLIF import FirstOrderFractionalLIF

class AnalyticalLIF:
    def __init__(self, tau_rc=0.2, tau_ref=0.002, v_init=0, v_th=1): # Default values for tau_rc and v_init
        self.tau_rc = tau_rc # rate of decay
        self.tau_ref = tau_ref # refractory period
        self.v = v_init # initial potential
        self.v_th = v_th # threshold potential

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

        if self.v >= self.v_th: # Voltage is above the threshold
            spike_time = actual_time + self.tau_rc * math.log((self.v - I) / (self.v_th - I)) 
            self.refractory_time = self.tau_ref + spike_time - actual_time
        
            self.output = 1 / T_step                         # Fire
            self.v = 0                                       # Reset potential
        else:
            self.output = 0          # Don't fire

        return self.output

    # reset neurone to initial state
    def reset(self):
        self.output = self.refractory_time = 0
        self.v = self.v_init

def benchmark_neurones():
    """
        Compares FirstOrderLIF
                 FirstOrderFractionalLIF
                 AnalyticalLIF
        
        on accuracy of firing rate for various T_step sizes 
    """
    NUM_SECONDS = 2
    TIME_STEP_SIZES = []

    currSize = 0.0001
    while currSize < 0.002:
        TIME_STEP_SIZES.append(currSize)
        currSize *= 1.2

    def getSpikeCount(neuron, I, numSeconds, timeStepSize):
        numSteps = int(numSeconds / timeStepSize)
        count = 0
        for i in range(numSteps):
            output = neuron.step(I(i * timeStepSize), timeStepSize)
            if output > 0:
                count += 1
        return count

    neurons = {
        'analytical': lambda: AnalyticalLIF(),
        'firstOrder': lambda: FirstOrderLIF(),
        'firstOrderFractional': lambda: FirstOrderFractionalLIF()
    }

    avgErrors = { k: [] for k in neurons.keys() }
    errors = { k: {} for k in neurons.keys() }

    for timeStepSize in TIME_STEP_SIZES:
        for k in errors:
            errors[k][timeStepSize] = []

    for i in [1.2, 2, 4, 8, 16, 32, 64, 128]:
        I = lambda t: i
        groundTruthCount = getSpikeCount(AnalyticalLIF(), I, NUM_SECONDS, 1e-6)

        for neuronType in neurons:
            for timeStepSize in TIME_STEP_SIZES:
                neuron = neurons[neuronType]()
                err = abs(groundTruthCount - getSpikeCount(neuron, I, NUM_SECONDS, timeStepSize))
                errors[neuronType][timeStepSize].append(err)
                
            
    for neuronType in neurons:
        for timeStepSize in TIME_STEP_SIZES:
            avgErrors[neuronType].append(np.mean(errors[neuronType][timeStepSize]))

    # Plotting the error vs time step size
    plt.figure()

    for neuronType in neurons:
        plt.plot(TIME_STEP_SIZES, avgErrors[neuronType], label=neuronType)

    plt.xscale('log')
    plt.xlabel('Time Step Size')
    plt.ylabel('Average Error')
    plt.title('Error vs Time Step Size')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    benchmark_neurones()