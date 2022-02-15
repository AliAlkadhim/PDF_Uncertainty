import numpy as np
import matplotlib.pyplot as plt
import h5py
import emcee
# l = np.load('MCMC_samples.npy')
# print(l.shape)
# plt.hist(l[:,0])
# plt.show()

filename = 'SAMPLES.h5'

sampler = emcee.backends.HDFBackend(filename)

fig, ax = plt.subplots(2, 1, sharex=True)
for i in [0, 1]:
    ax[i].plot(sampler.chain[0,:,i], 'k-', lw=0.2)
    ax[i].plot([0, n_steps-1], 
             [sampler.chain[0,:,i].mean(), sampler.chain[0,:,i].mean()], 'r-')

ax[1].set_xlabel('sample number')
ax[0].set_ylabel('r')
ax[1].set_ylabel('p')

plt.show()