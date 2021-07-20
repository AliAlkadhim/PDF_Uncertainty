dirs=[]
for i in range(999):
    dirs.append('run_{}'.format(i))
#dirs


import os
import re; import pandas as pd
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
#         #print(i)
         params.append(float(value))
    for error in errors.split():
        if error =='constant':
            error_list.append(0.0)
        else:
            error_list.append(float(error))
    
#         sampled_param = np.random.uniform(low = j-error, high=j+error)
# means=np.array(means).astype(float)
        
#print(means)
    #print(values)
    #print(words[2].split())
    #values= words[2].split()[0]
    #print(values)4
    #for ind, word in enumerate(split_line):
    #print(split_line)
        #print(word)
#     with open('minuit.oin.txt', 'r') as second:
#         split_line
#         second.write()
    
    
infile.close()
params = np.asarray(params); error_list =np.asarray(error_list)

#############GENERATE PARAMETERS
generated_params=[]
for i in range(len(params)):
    param, error = params[i], error_list[i]
    generated_param = np.random.uniform(low = param-error, high=param+error)
    generated_params.append(generated_param)



    
    
    
print( params, generated_params, '\n\n', len(params), len(generated_params))


import numpy as np
MVN = np.load('MVN_1000.npy')
MVN


get_ipython().getoutput("pwd")


MVN[:,13][0]


MVN.shape


for i in range(13):
    print(MVN[:,i][0])


import os
os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')




import os; import subprocess as sp;from shutil import copyfile
dirs=[]
for i in range(1000):
    dirs.append('run_{}'.format(i))
#dirs

os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')
steering = 'steering.txt'
for run_ind, run in enumerate(dirs):
    os.makedirs(run, exist_ok=True)
    print(os.path.abspath(run))
    path=os.path.abspath(run)
    copyfile('ewparam.txt', os.path.join(path, 'ewparam.txt'))
    copyfile('steering.txt', os.path.join(path, 'steering.txt'))
    minuit_in_path = os.path.join(path, 'minuit.in.txt')
    with open(minuit_in_path, 'w') as second:
        #second = os.path.abspath(second)
        second.write('set title\n')
        second.write('new  14p HERAPDF\n')
        second.write('parameters\n')
        #lets put 0 for the fourth column, meaning that this parameter is fixed
        second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(MVN[:,0][run_ind])+ '    '+'0.\n')
        second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(MVN[:,1][run_ind])+ '    '+'0.\n')
        second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(MVN[:,2][run_ind])+ '    '+'0.\n')
        second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(MVN[:,3][run_ind])+ '    '+'0.\n')
        second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(25.000)+ '    '+'0.\n')
        #note that Cprig is a constant, not a parameter value!
        second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(MVN[:,4][run_ind])+ '    '+'0.\n')
        second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(MVN[:,5][run_ind])+ '    '+'0.\n')
        second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(MVN[:,6][run_ind])+ '    '+'0.\n')
        second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(MVN[:,7][run_ind])+ '    '+'0.\n')
        second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(MVN[:,8][run_ind])+ '    '+'0.\n')
        second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(MVN[:,9][run_ind])+ '    '+'0.\n')
        second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(MVN[:,10][run_ind])+ '    '+'0.\n')
        second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(MVN[:,11][run_ind])+ '    '+'0.\n')
        second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(MVN[:,12][run_ind])+ '    '+'0.\n')
        second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(MVN[:,13][run_ind])+ '    '+'0.\n')
        second.write('\n\n\n')
        #for complete fit, do
#         second.write('migrad 200000\n')
#         second.write('hesse\n')
#         second.write('set print 3\n\n')
        #to run only 3 iterations, do 
        second.write('call fcn 2\n')
        second.write('*migrad 200000\n')
        second.write('*hesse\n')
        second.write('set print 3\n\n')
        second.write('return')
    os.chdir(path)
    sp.run('ln -s /home/ali/Desktop/Research/xfitter/xfitter-2.0.1/datafiles', shell=True)
    sp.run('xfitter', shell=True)
    os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')

    #sp.run('cp ./ewparam.txt run', shell=True)




import os; import subprocess as sp;from shutil import copyfile
dirs=[]
for i in range(1000):
    dirs.append('run_{}'.format(i))
#dirs

os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')
steering = 'steering.txt'
for run_ind, run in enumerate(dirs):
    os.makedirs(run, exist_ok=True)
    print(os.path.abspath(run))
    path=os.path.abspath(run)
    copyfile('ewparam.txt', os.path.join(path, 'ewparam.txt'))
    copyfile('steering.txt', os.path.join(path, 'steering.txt'))
    minuit_in_path = os.path.join(path, 'minuit.in.txt')
    with open(minuit_in_path, 'w') as second:
        #second = os.path.abspath(second)
        second.write('set title\n')
        second.write('new  14p HERAPDF\n')
        second.write('parameters\n')
        #lets put 0 for the fourth column, meaning that this parameter is fixed
        second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(MVN[:,0][run_ind])+ '    '+'0.\n')
        second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(MVN[:,1][run_ind])+ '    '+'0.\n')
        second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(MVN[:,2][run_ind])+ '    '+'0.\n')
        second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(MVN[:,3][run_ind])+ '    '+'0.\n')
        second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(25.000)+ '    '+'0.\n')
        #note that Cprig is a constant, not a parameter value!
        second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(MVN[:,4][run_ind])+ '    '+'0.\n')
        second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(MVN[:,5][run_ind])+ '    '+'0.\n')
        second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(MVN[:,6][run_ind])+ '    '+'0.\n')
        second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(MVN[:,7][run_ind])+ '    '+'0.\n')
        second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(MVN[:,8][run_ind])+ '    '+'0.\n')
        second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(MVN[:,9][run_ind])+ '    '+'0.\n')
        second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(MVN[:,10][run_ind])+ '    '+'0.\n')
        second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(MVN[:,11][run_ind])+ '    '+'0.\n')
        second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(MVN[:,12][run_ind])+ '    '+'0.\n')
        second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(MVN[:,13][run_ind])+ '    '+'0.\n')
        second.write('\n\n\n')
        #for complete fit, do
#         second.write('migrad 200000\n')
#         second.write('hesse\n')
#         second.write('set print 3\n\n')
        #to run only 3 iterations, do 
        second.write('call fcn 3\n')
        second.write('*migrad 200000\n')
        second.write('*hesse\n')
        second.write('set print 3\n\n')
        second.write('return')

    
for run_ind, run in enumerate(dirs):
    os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')
    path=os.path.abspath(run)
    os.chdir(path)
    sp.run('ln -s /home/ali/Desktop/Research/xfitter/xfitter-2.0.1/datafiles', shell=True)
    sp.run('xfitter', shell=True)
    #sp.run('cp ./ewparam.txt run', shell=True)







# import os; import subprocess as sp;from shutil import copyfile
# steering = 'steering.txt'
# for run in dirs:
#     os.makedirs(run, exist_ok=True)
#     print(os.path.abspath(run))
#     path=os.path.abspath(run)
#     copyfile('ewparam.txt', os.path.join(path, 'ewparam.txt'))
#     copyfile('steering.txt', os.path.join(path, 'steering.txt'))
#     minuit_in_path = os.path.join(path, 'minuit.in.txt')
#     with open(minuit_in_path, 'w') as second:
#         #second = os.path.abspath(second)
#         second.write('set title\n')
#         second.write('new  14p HERAPDF\n')
#         second.write('parameters\n')
#         #lets put 0 for the fourth column, meaning that this parameter is fixed
#         second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(generated_params[0])+ '    '+'0.\n')
#         second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(generated_params[1])+ '    '+'0.\n')
#         second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(generated_params[2])+ '    '+'0.\n')
#         second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(generated_params[3])+ '    '+'0.\n')
#         second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(generated_params[4])+ '    '+'0.\n')
#         second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(generated_params[5])+ '    '+'0.\n')
#         second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(generated_params[6])+ '    '+'0.\n')
#         second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(generated_params[7])+ '    '+'0.\n')
#         second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(generated_params[8])+ '    '+'0.\n')
#         second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(generated_params[9])+ '    '+'0.\n')
#         second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(generated_params[10])+ '    '+'0.\n')
#         second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(generated_params[11])+ '    '+'0.\n')
#         second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(generated_params[12])+ '    '+'0.\n')
#         second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(generated_params[13])+ '    '+'0.\n')
#         second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(generated_params[14])+ '    '+'0.\n')
#         second.write('\n\n\n')
#         second.write('migrad 200000\n')
#         second.write('hesse\n')
#         second.write('set print 3\n\n')
#         second.write('return')

#     #sp.run('cp ./ewparam.txt run', shell=True)


weights=[]
dirs=[]
for i in range(79):
    dirs.append('run_{}'.format(i))
    
os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')
for run_ind, run in enumerate(dirs):
    run_path = os.path.abspath(run)
    output_path = os.path.join(run_path, 'output')
    os.chdir(output_path)
    #print(os.getcwd())
    with open('Results.txt', 'r') as f:
        lines = f.readlines()
        chi2_line = lines[12]
        chi2 = chi2_line.split()[2]
        weights.append(float(chi2))
    os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories')



weights


first_fit=MVN[:,0:13][0]; first_weight=weights[0]
print('the first fit parameter values are: ', first_fit, '\nthe associated weight for this fit is', first_weight)
print('hence the number of fits has to be the same as the number of weights')


import matplotlib.pyplot as plt
Bg = first_fit=MVN[:,0][0:79]
plt.hist(Bg, bins=10, weights=weights)


import re

os.chdir('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/make_MVN_directories/run_1/output')
pattern  = re.compile(r'Correlated Chi2')
#we can use re to find the ch^2, or we can see that the correlated chi2 is lways on the 13th line!
with open('Results.txt', 'r') as f:
    lines = f.readlines()
    chi2_line = lines[12]
    chi2 = chi2_line.split()[2]
    #lines = f.read()

    #matches = pattern.finditer(lines)
        
    print(chi2)


import seaborn as sns
colors=sns.color_palette("rocket",3)
# sns.set_style("white")

# sns.set_context("poster")
plt.style.use('seaborn-paper')

fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(10,15))
axes[0,0].hist(MVN[:,0][0:79],bins=15, label='Bg',weights=weights)
#axes[0,0].set(title='Bg', xlabel='value')
axes[0,1].hist(MVN[:,1][0:79],bins=15, label='Cg',weights=weights)
axes[0,2].hist(MVN[:,2][0:79],bins=15,label='Aprig',weights=weights)
axes[0,3].hist(MVN[:,3][0:79],bins=15, label='Buv',weights=weights)
axes[1,0].hist(MVN[:,4][0:79],bins=15, label='Cuv',weights=weights)
axes[1,1].hist(MVN[:,5][0:79],bins=15,label='Euv',weights=weights)
axes[1,2].hist(MVN[:,6][0:79],bins=15, label='Bdv',weights=weights)
axes[1,3].hist(MVN[:,7][0:79],bins=15, label='Cdv',weights=weights)
axes[2,0].hist(MVN[:,8][0:79],bins=15, label='CUbar',weights=weights)
axes[2,1].hist(MVN[:,9][0:79],bins=15,label='DUbar',weights=weights)
axes[2,2].hist(MVN[:,10][0:79],bins=15,label='ADbar',weights=weights)
axes[2,3].hist(MVN[:,11][0:79],bins=15,label='BDbar',weights=weights)
axes[3,0].hist(MVN[:,12][0:79],bins=15,label='CDbar',weights=weights)
axes[3,1].hist(MVN[:,13][0:79],bins=15,label='CDbar',weights=weights)
axes[3,2].hist(MVN[:,13][0:79],bins=15,label='CDbar',weights=weights)
axes[3,3].hist(MVN[:,13][0:79],bins=15,label='CDbar',weights=weights)
plt.tight_layout(); plt.suptitle('HERAPDF Weighted Parameters')
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


get_ipython().getoutput("pwd")


chi2 = []
# for run_ind, run in enumerate(dirs):
    





Cg = MVN[:,1]
Cg


Cg = MVN[:,1][0]
Cg


for c in Cg:
    print(c)



