import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.stats import multivariate_normal
# import seaborn as sns
# import pymc3 as pm
import arviz as az
import re
import subprocess as sb 
import os
import emcee


plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
mp.rc('text', usetex=True)
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#%matplotlib inline

#LOAD DATA
chi2_array_ALL_DATA_25k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
MVN_25k_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')

#init_params = params_MASTER
#sb.run("source /home/ali/Desktop/Research/xfitter/xfitter_master_version/setup.sh", shell =True)
path = os.getcwd()
chi2_vals =[]

def ll(params):

    minuit_in_path = os.path.join(path, 'parameters.yaml')
    with open(minuit_in_path, 'w') as second:
        second.write('Minimizer: MINUIT # CERES \n')
        second.write('MINUIT:\n')
        second.write('  Commands: | \n')
        second.write('    call fcn 1\n')
        second.write('    set str 2\n')
        second.write('    call fcn 3\n')
        second.write('\n')
        second.write('Parameters:\n')
        second.write('  Ag   :  DEPENDENT\n')
        second.write('  Adbar   : [ ' + str(float(params[0])) + ', 0. ]\n')
        second.write('  Agp   : [ ' + str(format(float(params[1]), '.6f')) + ', 0. ]\n')
        second.write('  Bdbar   : [ ' + str(format(float(params[2]), '.6f')) + ', 0. ]\n')
        second.write('  Bdv   : [ ' + str(format(float(params[3]), '.6f')) + ', 0. ]\n')
        second.write('  Cgp   : [ ' + str(25.000) + ', 0. ]\n')
        #note that Cprig is a constant, not a parameter value!
        second.write('  Auv  :  DEPENDENT\n')
        second.write('  Bg   : [ ' + str(format(float(params[4]), '.6f')) + ', 0. ]\n')
        second.write('  Bgp   : [ ' + str(format(float(params[5]), '.6f')) + ', 0. ]\n')
        second.write('  Duv  : [    0     ]\n')
        second.write('  Buv   : [ ' + str(format(float(params[6]), '.6f')) + ', 0. ]\n')
        second.write('  Adv  :  DEPENDENT\n')
        second.write('  Cdbar   : [ ' + str(format(float(params[7]), '.6f')) + ', 0. ]\n')
        second.write('  Cdv   : [ ' + str(format(float(params[8]), '.6f')) + ', 0. ]\n')
        second.write('  Aubar: [ 0.0, 0.0 ]\n')
        second.write('  Bubar: [ 0.0, 0.0  ]\n')
        second.write('  Cg   : [ ' + str(format(float(params[9]), '.6f')) + ', 0. ]\n')
        second.write('  Cubar   : [ ' + str(format(float(params[10]), '.6f')) + ', 0. ]\n')
        second.write('  Cuv   : [ ' + str(format(float(params[11]), '.6f')) + ', 0. ]\n')
        second.write('  Dubar   : [ ' + str(format(float(params[12]), '.6f')) + ', 0. ]\n')
        second.write('  Euv   : [ ' + str(format(float(params[13]), '.6f')) + ', 0. ]\n')
        second.write('\n')

        second.write('  ZERO : [ 0. ]\n')        
        second.write('  fs   :   [ 0.4, 0.0 ]\n')
        #second.write('  DbarToS: \"=fs/(1-fs)\"\n')

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
        second.write('    expression: \"Adbar*fs/(1-fs)*(x^Bdbar*(1-x)^Cdbar)\"\n')
        # second.write('    input: par_dbar\n')
        second.write('\n')
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
        # second.write('  proton-APFELff:\n')
        # second.write('    ? !include evolutions/APFEL.yaml\n')
        # second.write('    decomposition: proton\n')

        second.write('  proton-QCDNUM:\n')
        second.write('    ? !include evolutions/QCDNUM.yaml\n')
        second.write('    decomposition: proton\n')

        # second.write('  antiproton:\n')
        # second.write('    class: FlipCharge\n')
        # second.write('  neutron:\n')
        # second.write('    class: FlipUD\n')   

        # second.write('  proton-LHAPDF:\n')
        # second.write('    class: LHAPDF\n')
        # second.write('    set: \"NNPDF30_nlo_as_0118\"\n')
        # second.write('    member: 0\n')
        second.write('\n')

        second.write('Q0 : 1.378404875209\n')
        second.write('\n')
        second.write('? !include constants.yaml\n')
        second.write('\n')
        second.write('alphas : 0.118\n')
        second.write('\n')
        second.write('byReaction:\n')
        second.write('\n')        
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
        # second.write('  AFB:\n')
        # second.write('    ? !include reactions/AFB.yaml \n')
        second.write('  APPLgrid:\n')
        second.write('    ? !include reactions/APPLgrid.yaml\n')
        second.write('  Fractal_DISNC:\n')
        second.write('    ? !include reactions/Fractal_DISNC.yaml\n')
        second.write('\n\n')
        second.write('hf_scheme_DISNC :\n')
        second.write('  defaultValue : \'RT_DISNC\' \n')
        second.write('\n')
        second.write('hf_scheme_DISCC :\n')
        second.write('  defaultValue : \'BaseDISCC\' \n')
        second.write('\n')


    #this should still work because we are not minimizing the chi2, we are just inputting the fixed parameter values in the minuit.in.txt format
    #RUN XFITTER AND PIPE ITS OUTPUT TO S
    s = os.popen("xfitter").read()
    #this executes the command xfitter, and the command that will go to the screen will be captured here

    #THIS IS THE EXPRESSION FORM: 
    # @chi2out__   503.08321706305105     

    #pattern = re.compile('[@chi2out].[0-9]+[.][0-9]+')
    pattern = re.compile('[@chi2out].[\d+]+[.][\d+]+'); regex=r'@chi2out__...\d+\.\d+'
    #matches = pattern.finditer(s)
    matches = re.findall(regex, s, re.MULTILINE)
    #print(matches)

    chi2 = matches[0].split()[1]
    f = -0.5 * float(chi2)
    
    if np.isnan(f):
        return - 1e3
    else:
        return f
 

# ndim=14
# nwalkers =1

nparams=14
niter    = 10 # number of iterations
nwalkers = 28

init_params = np.random.randn(nwalkers, nparams)
#p0 = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, nparams, ll)

niter=2

state = sampler.run_mcmc(init_params,niter, progress=True)
#name='likelihood.db'
#jb.dump(sample, name)
#sampler.reset()
#100 is the burn in 
#10000 steps

#samples = sampler.get_chain(flat=True)
ndiscard = 1
nthin    = 1
sample   = sampler.get_chain(discard=ndiscard, 
                             thin=nthin, 
                             flat=True)
print(sample)
