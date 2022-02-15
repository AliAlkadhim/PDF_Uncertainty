import numpy as np
k = np.load('MVN_25k_MASTER.npy')

print(np.mean(k[:,0]), np.mean(k[:,1]), np.mean(k[:,2]), np.mean(k[:,3]), np.mean(k[:,4]))
