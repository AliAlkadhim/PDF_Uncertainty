import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.stats import multivariate_normal
import seaborn as sns
import pymc3 as pm
import arviz as az

plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
mp.rc('text', usetex=True)
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

get_ipython().run_line_magic("matplotlib", " inline")


#LOAD DATA
chi2_array_ALL_DATA_25k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
MVN_25k_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')


COV_MASTER


MVN_25k_MASTER.shape


added = np.ones((1,14))
m = np.vstack((MVN_25k_MASTER, added))
m[-1]


g = np.random.multivariate_normal(params_MASTER, COV_MASTER, 1)
g


float(g[:,2])


samples = np.empty((3,14))
s1=samples[1]
str(float(s1[2]))


samples[0][0]



