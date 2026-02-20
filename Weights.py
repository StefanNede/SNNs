# Example of using weights to connect multiple layers

import numpy as np
from LIFCollection import LIFCollection
from SynapseCollection import SynapseCollection
t_step = 0.001

neurons_a = LIFCollection(n=50, tau_rc=0.02, tau_ref=0.002, T_step=t_step)
synapses_a = SynapseCollection(n=neurons_a.n, tau_s=0.1, T_step=t_step)
neurons_b = LIFCollection(n=40, tau_rc=0.02, tau_ref=0.002, T_step=t_step)
synapses_b = SynapseCollection(n=neurons_b.n, tau_s=0.1, T_step=t_step)

weights = np.random.randn(neurons_a.n, neurons_b.n)

outp = []
def step(inp):
    a  = neurons_a.step(inp)
    b  = synapses_a.step(a)
    bw = b @ weights
    c  = neurons_b.step(bw)
    d  = synapses_b.step(c)
    return (a, b, bw, c, d)

T = 10

times = np.arange(0, T, t_step)
def inp(t):
    return np.sin(t)

for t in times:
    outp.append(step(inp(t)))