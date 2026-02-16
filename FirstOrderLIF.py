# T_step is going to be a global variable 
# I will be defined on function calls to step()
import numpy as np
import matplotlib.pyplot as plt
from math import floor

T_step = 0.001 # Time step size
duration = 4.0   # Duration of the simulation: 4 seconds

class FirstOrderLI:
    def __init__(self, tau_rc=0.2, v_init=0):
        self.tau_rc = tau_rc
        self.v = v_init

    # advance 1 time step 
    def step(self, I, T_step): 
        leak_factor = T_step/self.tau_rc
        self.v = self.v*(1-leak_factor) + I*leak_factor

# FirstOrderLI w/ Firing
class FirstOrderLIF:
    def __init__(self, tau_rc=0.2, tau_ref=0.002, v_init=0, v_th=1):
        self.tau_rc = tau_rc # rate of decay
        self.tau_ref = tau_ref # refractory period
        self.v = v_init # initial potential
        self.v_th = v_th # threshold potential

        self.output = 0
        self.refractory_time = 0

    # advance 1 time step and return output of neuron
    def step(self, I, T_step): 
        self.refractory_time -= T_step

        # can we accept input
        if self.refractory_time < 0:
            leak_factor = T_step/self.tau_rc
            self.v = self.v*(1-leak_factor) + I*leak_factor # integrate input potential

        # print(self.v, self.v_th)
        # fire if potential above threshold else output = 0
        if self.v >= self.v_th: 
            self.refractory_time = self.tau_ref
            self.output = 1/T_step
            self.v = 0
        else:
            self.output = 0
        
        return self.output


def graph_potential(times, v_history, graph_title="", filename=""):
    plt.figure() # Create a new figure
    plt.plot(times, v_history) # Plot the neuron's potential over time
    plt.xlabel('Time (s)') # Label the x-axis
    plt.ylabel('Neuron potential') # Label the y-axis
    plt.title(graph_title)
    if filename:
        plt.savefig(f"{filename}.png")
    plt.show() # Display the plot

def no_potential_no_fire():
    print("NO POTENTIAL GOING IN AND NOT FIRING")
    I = 0     # No input current
    neuron = FirstOrderLI(v_init = 0.6) 

    v_history = []
    times = np.arange(0, duration, T_step) # Create a range of time values
    for t in times: # Loop over time
        v_history.append(neuron.v) # Record the neuron's potential
        neuron.step(I, T_step) # Advance one time step

    print(v_history[:5], "...", v_history[-5:]) # Print the first and last 5 potentials
    graph_potential(times, v_history, graph_title="Leaky Neuron w/ No Input Potential", filename="no_potential_no_fire")
    print()

# spends first half of period at max amplitude and second half off
# period is integer, t is float
def square_wave_func(period, t, magnitude=1):
    return magnitude if t%period < period/2 else 0

def square_wave_input():
    print("SQUARE WAVE INPUT FUNCTION AND NOT FIRING")

    period = 2 # so 1s on and 1s off

    neuron = FirstOrderLI(tau_rc=0.2, v_init=0.6)
    v_history = []
    times = np.arange(0, duration, T_step) # Create a range of time values
    for t in times: # Loop over time
        v_history.append(neuron.v) # Record the neuron's potential
        neuron.step(square_wave_func(period, t), T_step) # Advance one time step

    print(v_history[:5], "...", v_history[-5:]) # Print the first and last 5 potentials
    graph_potential(times, v_history, graph_title="Leaky Neuron w/ Square Input Potential", filename="square_potential_no_fire")
    print()

def square_wave_input_fire():
    global duration
    print("SQUARE WAVE INPUT FUNCTION AND FIRING")

    neuron = FirstOrderLIF(tau_rc=0.1, tau_ref=0.2, v_init=0, v_th=2)
    duration = 6.0
    period = 2 # so 1s on and 1s off
    voltage_magnitude = 2.01 # when input is on with what magnitude does it come in

    I_history = []
    v_history = []
    output_history = []
    vth_history = []

    times = np.arange(1, duration, T_step) # Create a range of time values

    for t in times: # Iterate over each time step
        I = square_wave_func(period, t, voltage_magnitude)     # Get the input current at this time
        neuron.step(I, T_step) # Advance the neuron one time step

        I_history.append(I)    # Record the input current
        v_history.append(neuron.v) # Record the neuron's potential
        output_history.append(neuron.output * T_step / 10) # Record the neuron's output (scaled)
        vth_history.append(neuron.v_th) # Record the neuron's threshold


    plt.figure() # Create a new figure
    plt.plot(times, I_history, color="grey", linestyle="--")
    plt.plot(times, vth_history, color="green", linestyle="--")
    plt.plot(times, v_history)
    plt.plot(times, output_history, color="red", linewidth=2.5)
    plt.xlabel('Time (s)') # Label the x-axis
    plt.legend(['Input current', 'neuron.v_th', 'neuron.v', 'neuron.output']) # Add a legend
    plt.show() # Display the plot


if __name__ == "__main__":
    # no_potential_no_fire()
    # square_wave_input()
    square_wave_input_fire()