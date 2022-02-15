import h5py; import emcee; import matplotlib.pyplot as plt
MCMC_samples = h5py.File('SAMPLER_ALLDATA.h5', 'r')
filename='SAMPLER_ALLDATA.h5'
reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain(discard=20, flat=True)
print('the samples have shape: ', samples.shape)
plt.hist(samples[:,1], bins=50)
plt.show()
