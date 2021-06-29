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


mean = [0, 0]
cov = [[1, 0], [0, 100]]  # di
import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


#for line in lines[9:n]:
line0 = lines[9]
import re
re.split('      ', line0).lstrip()


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


h = 6.626*10**(-34)
E_d = 5*70
c=3*10**8
l=1*10**(-9)
E_photon = h*c /l 
n = E_d/E_photon
n



