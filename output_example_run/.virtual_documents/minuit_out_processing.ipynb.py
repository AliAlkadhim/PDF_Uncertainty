file_lines = []
for line in open('minuit.out.txt', 'r'):
    file_lines.append(line)
print('the number of lines is', len(file_lines) )
n=25
print('Prameters block:', file_lines[9:n])
parameter_lines = str(file_lines[9:n])
print('\n \n')
print(parameter_lines.split())


def main():
    file =open('minuit.out.txt', 'r')
    lines = file.readlines()
    str_lines = lines.strip().split()
    header, values = lines[104], lines[106:124]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    file.close()
    print(data_dict)
main()



import os
import re
filename = 'minuit.out.txt'
infile = open(filename, 'r')
lines = infile.readlines()
print(lines[104])
for line in lines[106:124]:
    #print(line)
    #words = line.strip().split('      ')#the delimeter is 6 spaces to separate the columns
    words = re.split(r"(?: '\s+)\s*", line.strip())
    #use re noncapture group, defined as (?:...)since we dont want the separators in our 
    #result.
    #print(words[0])
    #print(words[2].split())
    values= words[2].split()
    print(values)
    #for ind, word in enumerate(split_line):
    #print(split_line)
        #print(word)
#     with open('minuit.oin.txt', 'r') as second:
#         split_line
#         second.write()
    
    
infile.close()
#line.split(0)



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


import matplotlib.pyplot as plt
x_pos = np.arange(len(params))
widths = 0.45
plt.bar(x_pos, params, widths, label='parameters')
plt.bar(x_pos + widths, generated_uniform_params, widths, label='generated uniform parameters')
plt.xlabel('Parameter'); plt.ylabel('Parameter value')
plt.legend()



for line in lines[127:143]:
    rows = line.strip().split('\n')
    triang_rows = rows[0]
#     for row in triang_rows.split('\n'):

#     for row in triang_rows.strip().split('\s'):
    for row in triang_rows.strip().split('\s'):

        print(row, '\n')


COV = np.empty((14,14))
cov_list = [None]*14
cov_list[0] = [0.632E-03] 
cov_list[1] = [0.000872, 0.0117]
cov_list[2] =[-0.844E-03,-0.893E-03, 0.120E-02]
cov_list[3]=[-0.122E-03,-0.150E-03, 0.197E-03, 0.581E-04]


cov_list[4] =[0.937E-05, 0.223E-04, -0.833E-05, -0.198E-05, 0.287E-04]
cov_list[4]


COV[13] = np.array([0.349E-02,-0.135E-01,-0.437E-02,-0.589E-03, 0.215E-03, 0.131E-04,-0.652E-02, 0.240E-02, 0.188E-02,-0.218E-01, 0.363E-01, 0.110E-03, 0.182E-04, 0.370E+00])
COV[12] = np.array([-0.558E-05,-0.582E-06, 0.425E-05, 0.155E-05, 0.768E-06,-0.388E-07, 0.580E-05, 0.193E-05,-0.315E-05, 0.182E-04,-0.388E-04, 0.518E-05, 0.304E-05,0])
COV[11] = np.array([-0.774E-05, 0.139E-04, 0.497E-05, 0.232E-05, 0.327E-05,-0.318E-06, 0.436E-04, 0.570E-05, 0.243E-06, 0.168E-04,-0.739E-04, 0.156E-04,0,0])
COV[10]= np.array([-0.191E-02, 0.429E-02, 0.264E-02, 0.335E-03,-0.221E-03,-0.661E-03, 0.638E-02,-0.487E-03,-0.170E-02, 0.253E-01, 0.982E-01,0,0,0]) 
COV[9] = np.array([0.935E-03,-0.214E-02,-0.136E-02,-0.173E-03, 0.242E-03, 0.473E-03, 0.194E-03, 0.662E-03, 0.120E-03, 0.498E-01,0,0,0,0])
COV[8] = np.array([0.246E-03,-0.593E-03,-0.290E-03,-0.491E-04, 0.140E-04,-0.398E-03, 0.296E-02, 0.170E-02, 0.158E-01, 0,0,0,0,0 ])
COV[7] = np.array([0.510E-04, 0.855E-04,-0.644E-04,-0.101E-04,-0.111E-04,-0.975E-05,-0.446E-03, 0.540E-03,0,0,0,0,0,0])
COV[6] = np.array([0.165E-03, 0.298E-04,-0.274E-03,-0.198E-04,-0.574E-03, 0.421E-02, 0.755E-01,0,0,0,0,0,0,0])
COV[5] = np.array([0.460E-04, -0.829E-04, -0.565E-04, -0.914E-05, 0.403E-04, 0.861E-03,0,0,0,0,0,0,0,0] )
COV[4] = np.array([0.937E-05, 0.223E-04, -0.833E-05, -0.198E-05, 0.287E-04,0,0,0,0,0,0,0,0,0])
COV[3] = np.array([-0.122E-03,-0.150E-03, 0.197E-03, 0.581E-04,0,0,0,0,0,0,0,0,0,0])
COV[2]=np.array([-0.844E-03,-0.893E-03, 0.120E-02,0,0,0,0,0,0,0,0,0,0,0])
COV[1] = np.array([0.000872, 0.0117,0,0,0,0,0,0,0,0,0,0,0,0])
COV[0] = np.array([0.632E-03,0,0,0,0,0,0,0,0,0,0,0,0,0] )
for i in range(len(COV)):
    for j in range(len(COV[i])):
        COV[i][j] = COV[j][i]
COV


y = COV.view(type=np.matrix)
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
matprint(COV)


COV.shape


cov_diag = COV.diagonal()
np.sqrt(cov_diag)


params = [-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810]
means = np.array(params)
means.shape



d=COV.shape[0] #this has to be 13 since
n=100 #number of samples, could be anything

def get_mvn_samples(mu,cov,n,d):
    samples = np.zeros((n,d))
    for i in range(n):      
        samples[i,:] = np.random.multivariate_normal(mu, cov, 1)
    
    return samples


MVN_10 = get_mvn_samples(mu=means, cov=COV, n=10, d=d)
MVN_10


MVN_double = get_mvn_samples(mu=means, cov=COV, n=10000, d=d)
MVN_double


print(MVN[0,:], MVN[0,:].shape, MVN.shape)


MVN_double.dtype


MVN.dtype


from numpy import asarray
from numpy import savetxt
# define data
# save to csv file
savetxt('MVN.dat', MVN, delimiter=', ', header = str(MVN.shape[0]),  fmt='get_ipython().run_line_magic("1.5f',", " comments='')")
#the comments='' gets rid of the # in front of my number of lines for the header


# from numpy import loadtxt

# data = loadtxt('MVN.dat', delimiter=',', comments='')
# data


with open('ex.dat','w') as f:
    f.write(list(MVN[0,:]))


with open('MVN.dat', 'w') as f:
    for i in range(MVN.shape[0]):
        f.write(MVN[i, :])


get_ipython().getoutput("pwd")


MVN[:,13]


np.save('MVN_10.npy',MVN_10)


get_ipython().getoutput("pwd")


Bg = MVN[:,0]
plt.hist(Bg.flatten(), bins=50)


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


# data=[MVN[:,1],
# MVN[:,2],
# MVN[:,3],
# MVN[:,4],
# MVN[:,5],
# MVN[:,6],
# MVN[:,7],
# MVN[:,8],
# MVN[:,9],
# MVN[:,10],
# MVN[:,11],
# MVN[:,12],
# MVN[:,13]]

def f(z, μ, Σ):
    """
    The density function of multivariate normal distribution.

    Parameters
    ---------------
    z: ndarray(float, dim=2)
        random vector, N by 1
    μ: ndarray(float, dim=1 or 2)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1
    """

    z = np.array(z)
    μ = np.array(μ)
    Σ = np.array(Σ)

    N = z.size

    temp1 = np.linalg.det(Σ) ** (-1/2)
    temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))

    return (2 * np.pi) ** (-N/2) * temp1 * temp2

f(MVN[1,:], means, COV)


def my_mv_pdf(x, mu, cov):
    k = len(x)
    uu = mu.reshape(k,1)
    xx = x.reshape(k,1)
    t1 = (2*np.pi)**2
    t2 = np.linalg.det(cov)
    t3 = 1.0/np.sqrt(t1*t2)
    t4 =(xx-uu).T
    t5 = np.linalg.inv(cov)
    t6= (xx-uu)
    t7 = -0.5 *(np.dot(t4, t5).dot(t6))
    result = t3* np.exp(t7)
    return result
my_mv_pdf(MVN[1,:], means, COV)


class MultivariateNormal:
    """
    Class of multivariate normal distribution.

    Parameters
    ----------
    μ: ndarray(float, dim=1)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1

    Arguments
    ---------
    μ, Σ:
        see parameters
    μs: list(ndarray(float, dim=1))
        list of mean vectors μ1 and μ2 in order
    Σs: list(list(ndarray(float, dim=2)))
        2 dimensional list of covariance matrices
        Σ11, Σ12, Σ21, Σ22 in order
    βs: list(ndarray(float, dim=1))
        list of regression coefficients β1 and β2 in order
    """

    def __init__(self, μ, Σ):
        "initialization"
        self.μ = np.array(μ)
        self.Σ = np.atleast_2d(Σ)

    def partition(self, k):
        """
        Given k, partition the random vector z into a size k vector z1
        and a size N-k vector z2. Partition the mean vector μ into
        μ1 and μ2, and the covariance matrix Σ into Σ11, Σ12, Σ21, Σ22
        correspondingly. Compute the regression coefficients β1 and β2
        using the partitioned arrays.
        """
        μ = self.μ
        Σ = self.Σ

        self.μs = [μ[:k], μ[k:]]
        self.Σs = [[Σ[:k, :k], Σ[:k, k:]],
                   [Σ[k:, :k], Σ[k:, k:]]]

        self.βs = [self.Σs[0][1] @ np.linalg.inv(self.Σs[1][1]),
                   self.Σs[1][0] @ np.linalg.inv(self.Σs[0][0])]

    def cond_dist(self, ind, z):
        """
        Compute the conditional distribution of z1 given z2, or reversely.
        Argument ind determines whether we compute the conditional
        distribution of z1 (ind=0) or z2 (ind=1).

        Returns
        ---------
        μ_hat: ndarray(float, ndim=1)
            The conditional mean of z1 or z2.
        Σ_hat: ndarray(float, ndim=2)
            The conditional covariance matrix of z1 or z2.
        """
        β = self.βs[ind]
        μs = self.μs
        Σs = self.Σs

        μ_hat = μs[ind] + β @ (z - μs[1-ind])
        Σ_hat = Σs[ind][ind] - β @ Σs[1-ind][1-ind] @ β.T

        return μ_hat, Σ_hat
multi_normal = MultivariateNormal(means, COV)
multi_normal


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


COV = np.empty((14,14))
for i in range(len(COV)):
    cov_list_list[i]= cov_list[i]
#     for j in range(len(COV[i])):
#         COV[i][j] = cov_list_list[i][j]
            
#COV
cov_list_list


# for line in chain_cov_mat:
#     row = line.strip().split()
#     for i in range(14):
#         for row_val in row[i].split():
            
#             for j in range(14):
#                 for col_val in row[i][j].split():
#                     COV[row_val][col] = float(row[i][j])
cov_list=[]
COV = np.empty((14,14))

# delimeters = "-", " "
# regexPattern = '|'.join(map(re.escape, delimiters))
pattern = re.compile(r'[\s\S.\d\D\w\W]\d\.\d\d\d[E]-\d\d')
#matches = pattern.finditer(text_to_search)
# for match in matches:
#     print(match)
for line in lines[127:143]:
    rows = line.strip().split('\n')
    triang_rows = rows[0]
#     for row in triang_rows.split('\n'):

    for row in triang_rows.strip().split('\s'):

        matches = pattern.finditer(row)
        inner_list=[]
        for match in matches:
            inner_list.append(float(row[match.span()[0]:match.span()[1]]))
            #cov_list.append(float(row[match.span()[0]:match.span()[1]]))
        cov_list.append(inner_list)
#         split_row = re.split(regexPattern, row)
#         for val in split_row:
#             cov_list.append(float(val))
        
        #for val in row:
            
#     for value in value_0:
        
#         cov_list.append(float(value))
cov_list


cov_list[0] = [0.632e-03]
cov_list[1] = [0.872E-03, 0.117E-01]
cov_list[:4]


COV = np.empty((14,14))



len(cov_list)


length = max(map(len, cov_list))
cov = np.array([xi+[None]*(length-len(xi)) for xi in cov_list])
cov.shape


############WRITE GENERATED PARAMETERS INTO NEW MINUIT.IN FILE
with open('minuit_ex.in.txt', 'w') as second:
    second.write('set title\n')
    second.write('new  14p HERAPDF\n')
    second.write('parameters\n')
    #lets put 0 for the fourth column, meaning that this parameter is fixed
    second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(generated_params[0])+ '    '+'0.\n')
    second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(generated_params[1])+ '    '+'0.\n')
    second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(generated_params[2])+ '    '+'0.\n')
    second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(generated_params[3])+ '    '+'0.\n')
    second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(generated_params[4])+ '    '+'0.\n')
    second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(generated_params[5])+ '    '+'0.\n')
    second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(generated_params[6])+ '    '+'0.\n')
    second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(generated_params[7])+ '    '+'0.\n')
    second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(generated_params[8])+ '    '+'0.\n')
    second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(generated_params[9])+ '    '+'0.\n')
    second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(generated_params[10])+ '    '+'0.\n')
    second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(generated_params[11])+ '    '+'0.\n')
    second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(generated_params[12])+ '    '+'0.\n')
    second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(generated_params[13])+ '    '+'0.\n')
    second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(generated_params[14])+ '    '+'0.\n')
    second.write('\n\n\n')
    second.write('migrad 200000\n')
    second.write('hesse\n')
    second.write('hesse\n')
    second.write('set print 3\n\n')
    second.write('return')
#we dont have to close it since we are using a context manager "with open()"


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


filename = 'minuit.out.txt'
infile = open(filename, 'r')
lines = infile.readlines()
with open('minuit.in.txt', 'w') as second:
    for line in lines[9:n]:
        split_line = line.strip().split('      ')#the delimeter is 6 spaces to separate the columns
        for i in split_line:
            second.write(i)
infile.close()
second.close()
