'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script create a single neuron and inject it with current to see the membrane potential dynamics.
From sinabs documentation.
'''



import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('qt5agg')

lenstim = 100

ts = torch.zeros(1, lenstim, 1) + 1
x = np.zeros(lenstim).astype(int)
y = np.zeros(lenstim).astype(int)
p = np.zeros(lenstim).astype(int)


class RFs:
    def __init__(self, tau_mem):
        self.neuron = sl.LIF(tau_mem=tau_mem)
        self.vmem = []


width = 2
height = 2
num_neurons = width * height
tau_mem = 1
# create a layer of objects RFs neurons
neurons = [[RFs(tau_mem=tau_mem) for _ in range(width)] for _ in range(height)]

# Output spike raster
spike_train = []
i=0
for t in range (0, lenstim-1):
    with torch.no_grad():
        RF = neurons[y[i]][x[i]]
        out = RF.neuron(ts[:, i : i + 1])
        # Check if there is a spike
        if (out != 0).any():
            spike_train.append(ts.unsqueeze(dim=0))
        # Record membrane potential
        RF.vmem.append(RF.neuron.v_mem[0, 0])
    i += 1

# Plot membrane potential
plt.figure()
plt.plot(range(lenstim-1), neurons[0][0].vmem)
plt.title("LIF membrame dynamics")
plt.xlabel("$t$ [ms]")
plt.ylabel("$V_{mem}$");
plt.show()


print('end')

