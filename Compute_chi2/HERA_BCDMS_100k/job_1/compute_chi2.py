import os
import numpy as np
import re
MVN_100k = np.load('/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/RUNS/NNPDF_Uncertainty/Compute_chi2/MVN_samples/MVN_100k.npy')

num_samples=100000
path = os.getcwd()
#os.chdir(os.path.join(path, 'src/RUNS/example_batch_job'))
chi2_vals =[]

minuit_in_path = os.path.join(path, 'minuit.in.txt')
with open(minuit_in_path, 'w') as second:
    second.write('set title\n')
    second.write('new  14p HERAPDF\n')
    second.write('parameters\n')
    #lets put 0 for the fourth column, meaning that this parameter is fixed
    second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(MVN_100k[:,0][1])+ '    '+'0.\n')
    second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(MVN_100k[:,1][1])+ '    '+'0.\n')
    second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(MVN_100k[:,2][1])+ '    '+'0.\n')
    second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(MVN_100k[:,3][1])+ '    '+'0.\n')
    second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(25.000)+ '    '+'0.\n')
    #note that Cprig is a constant, not a parameter value!
    second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(MVN_100k[:,4][1])+ '    '+'0.\n')
    second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(MVN_100k[:,5][1])+ '    '+'0.\n')
    second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(MVN_100k[:,6][1])+ '    '+'0.\n')
    second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(MVN_100k[:,7][1])+ '    '+'0.\n')
    second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(MVN_100k[:,8][1])+ '    '+'0.\n')
    second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(MVN_100k[:,9][1])+ '    '+'0.\n')
    second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(MVN_100k[:,10][1])+ '    '+'0.\n')
    second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(MVN_100k[:,11][1])+ '    '+'0.\n')
    second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(MVN_100k[:,12][1])+ '    '+'0.\n')
    second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(MVN_100k[:,13][1])+ '    '+'0.\n')
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
s = os.popen('xfitter').read()
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
    with open('/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/RUNS/NNPDF_Uncertainty/Compute_chi2/HERA_BCDMS_100k/collected_chi2.txt', 'a') as f:
        f.write(chi2_val)
	f.write('\n')

#    chi2_vals.append(float(chi2_val))
#ith open('MVN_100k_chi2s.txt', 'w') as MVN_chi2:
#    for item in chi2_vals_1:
#    MVN_chi2.write(chi2_vals_1)
#chi2_array_100k = np.array(chi2_vals)
#np.save('chi2_array_100k.npy', chi2_array_100k)
#print(chi2_vals)
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

