import os
import re; import pandas as pd; import numpy as np
params=[]
generated_params = []
error_list=[]
filename = 'minuit.out.txt'
infile = open(filename, 'r')
lines = infile.readlines()
#print(lines[104])
# for line in lines[106:109, 111:120]:
from itertools import *
#make a chain iterator of our wanted lines from the output file, we want to exclude
#those lines that have "constant" for the error, since these are not the PDF parameters and we can't use them
chain = chain(islice(lines, 106, 111), islice(lines, 111, 121))
chain_cov_mat = islice(lines, 127, 143)

#for line in lines[106:121]:
for line in chain:
    #print(line)
    #words = line.strip().split('\s+')#the delimeter is 6 spaces to separate the columns
    words = line.strip().split()
    
    #words = re.split(r"(?: '\s*)\s*", line.strip())
    #df = pd.read_table(words)
    #use re noncapture group, defined as (?:...)since we dont want the separators in our 
    #result.
    #print(words[2])
    values= words[2]
    errors=words[3]
    #print(errors)
    for value in values.split():
        params.append(float(value))
        
    
    for error in errors.split():
        if error =='constant':
            #'constant' just means the parameter does not have error!
            error_list.append(0.0)
        else:
            error_list.append(float(error))
    

infile.close()
params = np.asarray(params); error_list =np.asarray(error_list)



#############GENERATE UNIFORM PARAMETERS
generated_uniform_params=[]
for i in range(len(params)):
    param, error = params[i], error_list[i]
    generated_param = np.random.uniform(low = param-error, high=param+error)
    generated_uniform_params.append(generated_param)

    
    
    
print( params, generated_uniform_params, '\n\n', len(params), len(generated_uniform_params))


param_labels = [1,4, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21]
param_names= ['Adbar', 'Agp', 'Bdbar', 'Bdv', 'Bg', 'Bgp', 'Buv', 'Cdbar', 'Cdv',  'Cg', 'Cubar', 'Cuv', 'Dubar', 'Euv' ]
param_values = np.array([0.16126, 0.73153E-01, -0.12732, 1.0301, -0.61872E-01, -0.38304, 0.81058, 9.5821, 4.8458, 5.5585, 7.0602, 4.8240, 1.5433, 9.9230 ])
print(r'$\vec{\theta}= $', param_values)
means=param_values


COV = np.empty((14,14))
cov_list = [None]*14
cov_list[0] = [0.445E-05] 
cov_list[1] = [0.344E-05, 0.209E-03]
cov_list[2] =[0.225E-05, 0.202E-05, 0.233E-05]
cov_list[3]=[-0.533E-05, -0.275E-03, -0.387E-05, 0.151E-04, 0.376E-03]


cov_list[4] =[0.128E-05, 0.514E-04, 0.107E-05, -0.265E-05, -0.576E-04, 0.333E-04]
cov_list[4]


COV[13] = np.array([0.250E-04,-0.193E-04, 0.576E-05,-0.376E-03, 0.235E-04, 0.708E-05,-0.398E-03,-0.488E-02, 0.256E-02,-0.629E-04, 0.203E-03, 0.302E-02, 0.439E-02, 0.568E-01])
COV[12] = np.array([-0.268E-04, 0.477E-03,-0.322E-04,-0.311E-03,-0.684E-03, 0.823E-04,-0.143E-03, 0.332E-01,-0.819E-03, 0.479E-02, 0.191E-01, -0.439E-03, 0.742E-01,0])
COV[11] = np.array([-0.982E-06,-0.142E-04,-0.239E-06, 0.149E-05, 0.210E-04,-0.369E-05, 0.312E-04, 0.441E-04,-0.326E-03,-0.604E-04, 0.262E-03, 0.675E-03,0,0])
COV[10]= np.array([0.541E-05,-0.185E-03, 0.164E-04, 0.452E-03, 0.255E-03,-0.266E-04, 0.159E-03,-0.205E-01,-0.270E-03,-0.275E-02, 0.379E-01,0,0,0]) 
COV[9] = np.array([-0.281E-05,-0.711E-03,-0.505E-05, 0.956E-04, 0.102E-02,-0.161E-03, 0.337E-04,-0.119E-01,-0.457E-03, 0.135E-01,0,0,0,0])
COV[8] = np.array([-0.241E-05,-0.926E-04,-0.332E-05, 0.125E-02, 0.137E-03,-0.234E-04, 0.204E-04, 0.488E-03, 0.131E-01, 0,0,0,0,0 ])
COV[7] = np.array([0.332E-04,-0.159E-02, 0.125E-04, 0.176E-02, 0.224E-02,-0.301E-03, 0.183E-03, 0.320E+00,0,0,0,0,0,0])
COV[6] = np.array([  0.123E-05, -0.397E-05, 0.389E-06, -0.905E-05, 0.679E-05,-0.147E-05, 0.217E-04,0,0,0,0,0,0,0])
COV[5] = np.array([  0.128E-05, 0.514E-04, 0.107E-05,-0.265E-05,-0.576E-04, 0.333E-04,0,0,0,0,0,0,0,0] )
COV[4] = np.array([-0.533E-05, -0.275E-03, -0.387E-05, 0.151E-04, 0.376E-03,0,0,0,0,0,0,0,0,0])
COV[3] = np.array([  0.227E-05, -0.884E-05, 0.128E-05, 0.430E-03,0,0,0,0,0,0,0,0,0,0])
COV[2]=np.array([0.225E-05, 0.202E-05, 0.233E-05,0,0,0,0,0,0,0,0,0,0,0])
COV[1] = np.array([0.344E-05, 0.209E-03,0,0,0,0,0,0,0,0,0,0,0,0])
COV[0] = np.array([0.445E-05,0,0,0,0,0,0,0,0,0,0,0,0,0] )
for i in range(len(COV)):
    for j in range(len(COV[i])):
        COV[i][j] = COV[j][i]
COV


COV.shape


cov_diag = COV.diagonal()
np.sqrt(cov_diag)



d=COV.shape[0] #this has to be 13 since
n=50000 #number of samples, could be anything

def get_mvn_samples(mu,cov,n,d):
    samples = np.zeros((n,d))
    for i in range(n):      
        samples[i,:] = np.random.multivariate_normal(mu, cov, 1)
    
    return samples


MVN_4000_MASTER = get_mvn_samples(mu=means, cov=COV, n=4000, d=d)
MVN_4000_MASTER


np.save('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy', MVN_4000_MASTER) 


MVN_50k = get_mvn_samples(mu=means, cov=COV, n=50000, d=d)
np.save('../Compute_chi2/MVN_samples/MVN_50k.npy', MVN_50k)
MVN_50k


MVN_25k = get_mvn_samples(mu=means, cov=COV, n=25000, d=d)
np.save('MVN_25k.npy', MVN_25k)
MVN_25k


print(MVN[0,:], MVN[0,:].shape, MVN.shape)


Chi2_vals = ['502.02258222162641', '505.51792996777914', '502.13814341380822', '504.73945700774379', '505.58203050662604', '507.07219476903686', '507.24231794975418', '503.19352301332844', '505.96248270647692', '506.83495056397925']
Chi2_array = np.array(Chi2_vals)
Chi2_array


MVN_10_chi2 = np.load('chi2_array.npy')
MVN_10_chi2


np.save('MVN_4000.npy',MVN_4000)


get_ipython().getoutput("pwd")


Cg = MVN[:,1]
     plt.hist(Cg.flatten(), bins=50)


for i in range(13):
    plt.hist(MVN[:,i], bins=100)


import seaborn as sns
colors=sns.color_palette("rocket",3)
# sns.set_style("white")

# sns.set_context("poster")
plt.style.use('seaborn-paper')

fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(10,15))
axes[0,0].hist(MVN[:,0],bins=100, label='Bg',weights=np.array([91]*10000))
#axes[0,0].set(title='Bg', xlabel='value')
axes[0,1].hist(MVN[:,1],bins=100, label='Cg',weights=np.array([91]*10000))
axes[0,2].hist(MVN[:,2],bins=100,label='Aprig',weights=np.array([91]*10000))
axes[0,3].hist(MVN[:,3],bins=100, label='Buv',weights=np.array([91]*10000))
axes[1,0].hist(MVN[:,4],bins=100, label='Cuv',weights=np.array([91]*10000))
axes[1,1].hist(MVN[:,5],bins=100,label='Euv',weights=np.array([91]*10000))
axes[1,2].hist(MVN[:,6],bins=100, label='Bdv',weights=np.array([91]*10000))
axes[1,3].hist(MVN[:,7],bins=100, label='Cdv',weights=np.array([91]*10000))
axes[2,0].hist(MVN[:,8],bins=100, label='CUbar',weights=np.array([91]*10000))
axes[2,1].hist(MVN[:,9],bins=100,label='DUbar',weights=np.array([91]*10000))
axes[2,2].hist(MVN[:,10],bins=100,label='ADbar',weights=np.array([91]*10000))
axes[2,3].hist(MVN[:,11],bins=100,label='BDbar',weights=np.array([91]*10000))
axes[3,0].hist(MVN[:,12],bins=100,label='CDbar',weights=np.array([91]*10000))
axes[3,1].hist(MVN[:,13],bins=100,label='CDbar',weights=np.array([91]*10000))
axes[3,2].hist(MVN[:,13],bins=100,label='CDbar',weights=np.array([91]*10000))
axes[3,3].hist(MVN[:,13],bins=100,label='CDbar',weights=np.array([91]*10000))
plt.tight_layout(); plt.suptitle('HERAPDF Parameters')
titles = ['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar','CDbar','CDbar','CDbar']
for i, ax in enumerate(axes.flatten()):
    ax.set(title=titles[i], xlabel='value')
    ax.legend()
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True, top=True)
# plt.tick_params(labelsize=14)
# plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
#plt.savefig('HERAPDF_params_MVN.png', dpi=300, bbox_inches='tight')
plt.show()


cor_Chi2 = 91.502780349924393     
Log_penalty_Chi2 = 5.8705183749659522 

import subprocess
subprocess.run('ls', capture_output=True)


dirs=[]
for i in range(10):
    dirs.append('dir_{}'.format(i))
dirs






# QQ Plot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

# generate univariate observations
data = MVN[:,0]
# q-q plot
qqplot(data, line='s')
qqplot(MVN[:,1], line='s')
plt.show()
qqplot()


5000*10/3600


fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(10,15))
plt.tight_layout(); plt.suptitle('Quantile-Quantile Plots for HERAPDF Parameters')

data=[MVN[:,1],
MVN[:,2],
MVN[:,3],
MVN[:,4],
MVN[:,5],
MVN[:,6],
MVN[:,7],
MVN[:,8],
MVN[:,9],
MVN[:,10],
MVN[:,11],
MVN[:,12],
MVN[:,13],
MVN[:,13],
MVN[:,13], MVN[:,13]]

for i, ax in enumerate(axes.flatten()):
    qqplot(data[i], ax = ax, line='s')
    ax.set(title=titles[i], xlabel='value')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
plt.savefig('QQPlots_HERAPDF_params_MVN.png', dpi=300, bbox_inches='tight')
plt.show()


MVN.mean(axis=0)


def get_weights(z, means, cov):
    z = np.array(z)
    
    mu = means # MVN.mean(axis=0)
    cov = np.array(cov)
    N = len(z)
    temp1 = np.linalg.det(cov) ** (-1/2)
    temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(cov) @ (z - mu))
    return (2 * np.pi) ** (-N/2) * temp1 * temp2

weights = [get_weights(z=MVN[i,:], means=MVN.mean(axis=0), cov=COV) for i in range(13)]
weights


np.at


MVN.shape


MVN[0,:]


means.shape


COV.shape


cov_list[5] = [0.460E-04, -0.829E-04, -0.565E-04, -0.914E-05, 0.403E-04, 0.861E-03] 
cov_list[6]= [0.165E-03, 0.298E-04,-0.274E-03,-0.198E-04,-0.574E-03, 0.421E-02, 0.755E-01]
cov_list[7] = [0.510E-04, 0.855E-04,-0.644E-04,-0.101E-04,-0.111E-04,-0.975E-05,-0.446E-03, 0.540E-03]
cov_list[8] =[0.246E-03,-0.593E-03,-0.290E-03,-0.491E-04, 0.140E-04,-0.398E-03, 0.296E-02, 0.170E-02, 0.158E-01 ]
cov_list[9] =[0.935E-03,-0.214E-02,-0.136E-02,-0.173E-03, 0.242E-03, 0.473E-03, 0.194E-03, 0.662E-03, 0.120E-03, 0.498E-01]
cov_list[10] = [-0.191E-02, 0.429E-02, 0.264E-02, 0.335E-03,-0.221E-03,-0.661E-03, 0.638E-02,-0.487E-03,-0.170E-02, 0.253E-01, 0.982E-01 ]
cov_list[11] = [-0.774E-05, 0.139E-04, 0.497E-05, 0.232E-05, 0.327E-05,-0.318E-06, 0.436E-04, 0.570E-05, 0.243E-06, 0.168E-04,-0.739E-04, 0.156E-04]
cov_list[12] = [-0.558E-05,-0.582E-06, 0.425E-05, 0.155E-05, 0.768E-06,-0.388E-07, 0.580E-05, 0.193E-05,-0.315E-05, 0.182E-04,-0.388E-04, 0.518E-05, 0.304E-05]
cov_list[13] = [0.349E-02,-0.135E-01,-0.437E-02,-0.589E-03, 0.215E-03, 0.131E-04,-0.652E-02, 0.240E-02, 0.188E-02,-0.218E-01, 0.363E-01, 0.110E-03, 0.182E-04, 0.370E+00]


cov_list


cov_list_list = [[None]*14]*14;


cov_list[0] = [0.632e-03]
cov_list[1] = [0.872E-03, 0.117E-01]
cov_list[:4]


COV = np.empty((14,14))



len(cov_list)


import itertools as IT
with open(filename, 'r') as f:
    lines = IT.chain(IT.islice(f, 0, 4), IT.islice(f, 5, 14) )
arr = np.genfromtxt(lines)


import numpy as np
#make a list of dtypes for each of the columns that we want

dtype1 = np.dtype([('NO.', 'int'), ('NAME', 'str'), ('VALUE', 'float32'), ('ERROR', 'float32')])
a = np.loadtxt(filename, dtype=dtype1, skiprows=106,  max_rows=4, usecols=(0, 1, 2, 3))
#np.loadtxt(filename, dtype)
a['VALUE']


words


import pandas as pd
df = pd.read_csv(filename, names=['NO','NAME','VALUE','ERROR'])[95:112]
#pd.read_csv(filename)ERROR
df.NO.apply(lambda x: pd.Series(str(x).split("\s+")))
#df.columns=['NO','NAME','VALUE','ERROR']


len(np.array(means))
means=np.array(means).astype(np.float)
means



def covariance_matrix(X):
    m = len(X) 
    mean = np.mean(X)
    cov_matrix = (X - mean).T.dot((X - mean)) / m-1
    np.random.seed(2020)
    return cov_matrix + 0.00001 
cov_mat_sig = np.array(covariance_matrix(means))
cov_mat_sig
