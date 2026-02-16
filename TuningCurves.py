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