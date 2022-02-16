import numpy as np
import h5py
import matplotlib.pyplot as plt


with h5py.File('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/MCMC_ALL_DATA_4k/SAMPLER_ALLDATA.h5','r') as hdf:
    ls = list(hdf.keys())
    print(ls)
    data = hdf.get('mcmc')
    dataset = data['chain']
    dataset = np.array(dataset)
dataset


dataset.shape


dataset[:,:,0].shape


dataset[0,:,0]


plt.hist(dataset[0,:,0].flatten(),bins=100)


plt.hist(dataset[0,:,8].flatten(),bins=100)


d1 = dataset[:,:,0].flatten()
print('shape',d1.shape)
plt.hist(dataset[:,:,0].flatten())


plt.hist(dataset[:,:,1].flatten())


plt.hist(dataset[:,:,5].flatten(), bins=100)


fig, ax = plt.subplots(14,1, figsize=(30,30))
for i in range(13):
    ax[i].hist(dataset[0,:,i].flatten(), bins=100)



