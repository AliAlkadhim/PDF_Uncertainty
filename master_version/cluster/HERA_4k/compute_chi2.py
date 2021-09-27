import os
import numpy as np
import re
MVN_4000 = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/MVN_samples/MVN_4000.npy')

num_samples=4000
path = os.getcwd()
chi2_vals =[]
for sample_ind in range(num_samples):
    minuit_in_path = os.path.join(path, 'parameters.yaml')
    with open(minuit_in_path, 'w') as second:
        second.write('Minimizer: MINUIT # CERES \n')
        second.write('MINUIT:\n')
        second.write('  Commands: | \n')
        second.write('    set str 2\n')
        second.write('    call fcn 3\n')
        second.write('\n')
        second.write('Parameters:\n')
        second.write('  Ag   :  DEPENDENT\n')
        second.write('  Bg   : [ ' + str(MVN_4000[:,0][sample_ind]) + ', 0. ]\n')
        second.write('  Cg   : [ ' + str(MVN_4000[:,1][sample_ind]) + ', 0. ]\n')
        second.write('  Agp   : [ ' + str(MVN_4000[:,2][sample_ind]) + ', 0. ]\n')
        second.write('  Bgp   : [ ' + str(MVN_4000[:,3][sample_ind]) + ', 0. ]\n')
        second.write('  Cgp   : [ ' + str(25.000) + ', 0. ]\n')
        #note that Cprig is a constant, not a parameter value!
        second.write('  Auv  :  DEPENDENT\n')
        second.write('  Buv   : [ ' + str(MVN_4000[:,4][sample_ind]) + ', 0. ]\n')
        second.write('  Cuv   : [ ' + str(MVN_4000[:,5][sample_ind]) + ', 0. ]\n')
        second.write('  Duv  : [    0     ]\n')
        second.write('  Euv   : [ ' + str(MVN_4000[:,6][sample_ind]) + ', 0. ]\n')
        second.write('  Adv  :  DEPENDENT\n')
        second.write('  Bdv   : [ ' + str(MVN_4000[:,7][sample_ind]) + ', 0. ]\n')
        second.write('  Cdv   : [ ' + str(MVN_4000[:,8][sample_ind]) + ', 0. ]\n')
        second.write('  Aubar: [ 0.0, 0.0 ]\n')
        second.write('  Bubar: [ 0.0, 0.0  ]\n')
        second.write('  Cubar   : [ ' + str(MVN_4000[:,9][sample_ind]) + ', 0. ]\n')
        second.write('  Dubar   : [ ' + str(MVN_4000[:,10][sample_ind]) + ', 0. ]\n')
        second.write('  Adbar   : [ ' + str(MVN_4000[:,11][sample_ind]) + ', 0. ]\n')
        second.write('  Bdbar   : [ ' + str(MVN_4000[:,12][sample_ind]) + ', 0. ]\n')
        second.write('  Cdbar   : [ ' + str(MVN_4000[:,13][sample_ind]) + ', 0. ]\n')
        second.write('\n')

        second.write('  ZERO : [ 0. ]\n')        
        second.write('  fs : [ 0.4, 0.0 ]\n')
        second.write('\n')

        second.write('Parameterisations:\n')
        second.write('  par_uv:\n')
        second.write('    class: HERAPDF\n')
        second.write('    parameters: [Auv,Buv,Cuv,Duv,Euv]\n')
        second.write('  par_dv:\n')
        second.write('    class: HERAPDF\n')
        second.write('    parameters: [Adv,Bdv,Cdv]\n')
        second.write('  par_ubar:\n')
        second.write('    class: HERAPDF\n')
        second.write('    parameters: [Adbar,Bdbar,Cubar,Dubar]\n')
        second.write('  par_dbar:\n')
        second.write('    class: HERAPDF\n')        
        second.write('    parameters: [Adbar,Bdbar,Cdbar]\n')
        second.write('  par_s:\n')
        second.write('    class: Expression\n')
        second.write('    expression: \"Adbar*fs/(1-fs)*(x^Bdbar*(1-x)^Cdbar)\" \n')
        second.write('  par_g:\n')
        second.write('    class: NegativeGluon\n')
        second.write('    parameters: [Ag,Bg,Cg,ZERO,ZERO,Agp,Bgp,Cgp]\n')
        second.write('\n\n')

        second.write('DefaultDecomposition: proton\n')
        second.write('Decompositions:\n')
        second.write('  proton:\n')
        second.write('    class: UvDvUbarDbarS\n')
        second.write('    xuv: par_uv\n')
        second.write('    xdv: par_dv\n')
        second.write('    xubar: par_ubar\n')
        second.write('    xdbar: par_dbar\n')
        second.write('    xs: par_s\n')
        second.write('    xg: par_g\n')
        second.write('\n')

        second.write('DefaultEvolution: proton-QCDNUM\n')
        second.write('\n\n')
        second.write('Evolutions:\n')
        second.write('  proton-QCDNUM:\n')
        second.write('    ? !include evolutions/QCDNUM.yaml\n')
        second.write('\n')
        second.write('Q0 : 1.378404875209\n')
        second.write('? !include constants.yaml\n')
        second.write('alphas : 0.118\n')
        second.write('byReaction:\n')
        second.write('  RT_DISNC:\n')
        second.write('    ? !include reactions/RT_DISNC.yaml\n')
        second.write('  FONLL_DISNC:\n')
        second.write('    ? !include reactions/FONLL_DISNC.yaml\n')
        second.write('  FONLL_DISCC:\n')
        second.write('    ? !include reactions/FONLL_DISCC.yaml\n')
        second.write('  FFABM_DISNC:\n')
        second.write('    ? !include reactions/FFABM_DISNC.yaml\n')
        second.write('  FFABM_DISCC:\n')
        second.write('    ? !include reactions/FFABM_DISCC.yaml\n')
        second.write('  AFB:\n')
        second.write('    ? !include reactions/AFB.yaml \n')
        second.write('  APPLgrid:\n')
        second.write('    ? !include reactions/APPLgrid.yaml\n')
        second.write('  Fractal_DISNC:\n')
        second.write('    ? !include reactions/Fractal_DISNC.yaml\n')
        second.write('\n\n')
        second.write('hf_scheme_DISNC :\n')
        second.write('  defaultValue : \'RT_DISNC\' \n')

        second.write('hf_scheme_DISCC :\n')
        second.write('  defaultValue : \'BaseDISCC\' \n')
        second.write('\n')
        second.write('WriteLHAPDF6:\n')   
        second.write('  name: \"proton\" \n')
        second.write('  description: \"...\" \n') 
        second.write('  authors: \"...\" \n')
        second.write('  reference: \"...\" \n') 
        second.write('  preferInternalGrid:\n')

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
    pattern = re.compile('[@chi2out].[\d+]+[.][\d+]+'); regex=r'After.minimisation....\d+\.\d+'
    #matches = pattern.finditer(s)
    matches = re.findall(regex, s, re.MULTILINE)
    #print(matches)
    
    for match in matches:
        chi2_val = match.split()[2]
        chi2_vals.append(chi2_val)
#ith open('MVN_10_chi2s.txt', 'w') as MVN_chi2:
#    for item in chi2_vals:
#    MVN_chi2.write(chi2_vals)
chi2_array_4000 = np.array(chi2_vals)
np.save('chi2_array_4000.npy', chi2_array_4000)
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