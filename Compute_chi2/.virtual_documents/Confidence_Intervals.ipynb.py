import os
import re
import numpy as np
import matplotlib.pyplot as plt


COV = np.load('../output_example_run/MVN_4000_COV.npy')
COV


params = np.load('../output_example_run/params_means.npy')
params


cov_diag = COV.diagonal()
sigmas = np.sqrt(cov_diag)


MVN_4000 = np.load('MVN_samples/MVN_4000.npy'); MVN_4000


Bg = MVN_4000[:,0]
print('The sample size=', len(Bg))
plt.hist(Bg.flatten(), bins=100); plt.title(r'$Bg$ unweighted distribtution')
xbar_Bg = Bg.mean()
print(r'$\bar{x}_{B_g}$' , xbar_Bg)


sigma_Bg = sigmas[0]; sigma_Bg


z_68, n = 0.7995, 4000
theta_hat_l_gauss_Bg = xbar_Bg - z_68*(sigma_Bg/np.sqrt(n))
theta_hat_l_gauss_Bg


theta_hat_u_gauss_Bg = xbar_Bg + z_68*(sigma_Bg/np.sqrt(n))
theta_hat_u_gauss_Bg









