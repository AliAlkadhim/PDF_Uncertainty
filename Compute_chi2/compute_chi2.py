import os
import numpy as np
import re
MVN = np.load('MVN_10.npy')

num_samples=10
path = os.getcwd()
chi2_vals =[]
for sample_ind in range(num_samples):
    minuit_in_path = os.path.join(path, 'minuit.in.txt')
    with open(minuit_in_path, 'w') as second:
        second.write('set title\n')
        second.write('new  14p HERAPDF\n')
        second.write('parameters\n')
        #lets put 0 for the fourth column, meaning that this parameter is fixed
        second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(MVN[:,0][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(MVN[:,1][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(MVN[:,2][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(MVN[:,3][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(25.000)+ '    '+'0.\n')
        #note that Cprig is a constant, not a parameter value!
        second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(MVN[:,4][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(MVN[:,5][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(MVN[:,6][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(MVN[:,7][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(MVN[:,8][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(MVN[:,9][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(MVN[:,10][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(MVN[:,11][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(MVN[:,12][sample_ind])+ '    '+'0.\n')
        second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(MVN[:,13][sample_ind])+ '    '+'0.\n')
        second.write('\n\n\n')
        #for complete fit, do
#         second.write('migrad 200000\n')
#         second.write('hesse\n')
#         second.write('set print 3\n\n')
        #to run only 3 iterations, do 
        second.write('call fcn 3\n')
        second.write('set print 3\n\n')
        second.write('return')
        
    
#this should still work because we are not minimizing the chi2, we are just inputting the fixed parameter values in the minuit.in.txt format
#RUN XFITTER AND PIPE ITS OUTPUT TO S
    s = os.popen("xfitter").read()
#this executes the command xfitter, and the command that will go to the screen will be captured here

# s= '''kasfoafjiop 
# @chi2out 505.735 
# iawshgoeihaoierg'''
# s = '@chi2out = 234.12'
    
#THIS IS THE EXPRESSION FORM: 
# @chi2out__   503.08321706305105     

#pattern = re.compile('[@chi2out].[0-9]+[.][0-9]+')
    pattern = re.compile('[@chi2out].[\d+]+[.][\d+]+'); regex=r'@chi2out__...\d+\.\d+'
    #matches = pattern.finditer(s)
    matches = re.findall(regex, s, re.MULTILINE)
    #print(matches)
    
    for match in matches:
        chi2_val = match.split()[1]
        chi2_vals.append(chi2_val)
print(chi2_vals)
#print(s)
#print(matches)
# for match in matches:
#     print(s[match.span()[0]:match.span()[1]])
# for line in s.strip().split('\n'):
#     match = pattern.finditer(line)
    # chi2val = float(line[match.span()[0]:match.span()[1]])
    # chi2_list.append(chi2val)
#pattern.findall(s)
#print(chi2val)