# T_step is going to be a global variable 
# I will be defined on function calls to step()
import numpy as np
import matplotlib.pyplot as plt
from math import floor

T_step = 0.001 # Time step size
duration = 4.0   # Duration of the simulation: 2 seconds

# TODO: add firing
class FirstOrderLIF:
    def __init__(self, tau_rc=0.2, v_init=0):
        self.tau_rc = tau_rc
        self.v = v_init

    # advance 1 time step 
    def step(self, I, T_step): 
        leak_factor = T_step/self.tau_rc
        self.v = self.v*(1-leak_factor) + I*leak_factor


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
def square_wave_func(period, t):
    return t%period <= period/2

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


if __name__ == "__main__":
    # no_potential_no_fire()
    square_wave_input()