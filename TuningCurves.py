import numpy as np
import matplotlib.pyplot as plt

from FirstOrderLIF import FirstOrderLIF

def getFiringRate(neuron, input_current, duration, T_step=0.001):
    """
        Get simulated spike/firing rate of a neuron given a constant input_current over 'duration' time period
    """
    spikes = 0
    numSteps = int(duration / T_step)

    for _ in range(numSteps):
        output = neuron.step(input_current, T_step)
        spikes += (output > 0)

    return spikes/duration

def getAnalyticalFiringRate(neuron, input_current):
    if input_current <= neuron.v_th:
        return 0
    else:
        return 1 / (neuron.tau_ref - neuron.tau_rc * np.log(1 - neuron.v_th/input_current))

# calculates decoder for a neuron -> minimises MSE between input signal and estimated signal
def computeDecoder(neuron, range_low=-1, range_high=1, interval=0.1):
    numerator   = 0
    denominator = 0
    for i in np.arange(range_low, range_high, interval):
        r = getAnalyticalFiringRate(neuron, i)
        numerator   += r*i
        denominator += r*r
    return numerator / denominator

# calculates col vectorof decoders for a collection of LIF 
def computeCollectionDecoder(lifs, range_low=-1, range_high=1, interval=0.01):
    inputs = np.arange(range_low, range_high, interval)

    # tuningCurves = computeTuningCurves(lifs, inputs)
    tuningCurves = []
    # build up row in matrix for neurone i
    for neurone, gain, bias, encoder in zip(lifs.neurones, lifs.gains, lifs.biases, lifs.encoders):
        tuningCurves.append([getAnalyticalFiringRate(neurone, i * gain * encoder + bias) for i in inputs])

    A = np.array(tuningCurves)

    value = np.expand_dims(inputs, axis=1)
    Gamma = A @ A.T + np.identity(len(lifs.neurones))
    GammaInv = np.linalg.inv(Gamma)
    Upsilon = A @ value

    Phi = GammaInv @ Upsilon

    return Phi

def plotTuningCurve():
    input_currents = np.arange(0,20.1,0.1)
    firing_rates = []
    analytical_rates = []

    for input_current in input_currents:
        neuron = FirstOrderLIF(tau_rc=0.3, tau_ref=0.2)
        firing_rate = getFiringRate(neuron, input_current, 10)
        analytical_rate = getAnalyticalFiringRate(neuron, input_current)
        firing_rates.append(firing_rate)
        analytical_rates.append(analytical_rate)
    
    # plot figure
    plt.figure()
    plt.plot(input_currents, firing_rates)
    plt.plot(input_currents, analytical_rates, color="C1", linestyle="--")
    plt.legend(["Simulated", "Analytical"])
    plt.xlabel("Input current")
    plt.ylabel("Firing rate")
    plt.title("Firing Rates vs Input Current")
    plt.show()


if __name__ == "__main__":
    plotTuningCurve()