import numpy as np
import h5py


with h5py.File('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/MCMC_ALL_DATA_4k/SAMPLER_ALLDATA.h5','r') as hdf:
    ls = list(hdf.keys())
    print(ls)



