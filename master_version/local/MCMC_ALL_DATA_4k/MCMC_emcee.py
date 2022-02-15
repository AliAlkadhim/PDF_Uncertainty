import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

import re
import subprocess as sb 
import os
import emcee
import h5py

plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
mp.rc('text', usetex=True)

#az.style.use("arviz-darkgrid")

#%matplotlib inline

#LOAD DATA
# chi2_array_ALL_DATA_25k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
# MVN_25k_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')

#init_params = params_MASTER
#RUN BELOW IN SHELL
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
        #second.write('    call fcn 3\n')
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

        second.write('  antiproton:\n')
        second.write('    class: FlipCharge\n')
        second.write('  neutron:\n')
        second.write('    class: FlipUD\n')   

        # second.write('  proton-LHAPDF:\n')
        # second.write('    class: LHAPDF\n')
        # second.write('    set: \"NNPDF30_nlo_as_0118\"\n')
        # second.write('    member: 0\n')
        # second.write('\n')

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
    #the above chi2 value is what we wexpect if we only use HERA data
    pattern = re.compile('[@chi2out].[\d+]+[.][\d+]+')
    regex=r'@chi2out__...\d+\.\d+'
    #matches = pattern.finditer(s)
    matches = re.findall(regex, s, re.MULTILINE)
    #print(matches)

    chi2 = matches[0].split()[1]
    f = -0.5 * float(chi2)
    
    if np.isnan(f):
        return - 1e3
    else:
        return f
 

############DEFINE PARAMETERS FOR OUR SAMPLER
ndim=14
nwalkers = 14*(2**10) #has to be even, preferabbly a multiple of n_params
nparams=14
n_burn    = 100 # "burn-in" period to let chains stabilize
n_steps = 50000   # number of MCMC steps to take after burn-in

#seed the random runmber generator
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


#Define initial starting values of the parameters.
# each walker gets a set of initial parameters, so the dimension of init_params=(nwalkers, n_params)
#below we could start the parameters in the vicinity of the best-fit values, which would reduce burn-in time
#init_params = np.random.randn(nwalkers, nparams)
def get_mvn_samples(mu,cov,n,d):
    samples = np.zeros((n,d))
    for i in range(n):      
        samples[i,:] = np.random.multivariate_normal(mu, cov, 1)
    
    return samples

init_params = get_mvn_samples(mu=params_MASTER, cov=COV_MASTER*10000, n=nwalkers, d=14)

#################SAVE THE SAMPLER TO ACCESS LATER: this requires setting a backend with h5py
filename = 'SAMPLER_ALLDATA.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
#you can access the sampler later (to check convergence, etc) by sampler = emcee.backends.HDFBackend(filename)
#Start the sampler. first arg=number of walkers, second: number of parameters, third the log likelihood, threads gives you th eoption to use more cores
sampler = emcee.EnsembleSampler(nwalkers, nparams, ll, backend = backend, threads=10)

#DO THE BIRN IN, RUN MCMC WITHOUT STORING THE RESULTS
pos, prob, state = sampler.run_mcmc(init_params, n_burn, progress=True)
sampler.reset()

# Sample again, starting from end burn-in state (starting at pos and ending at nsteops)
_ = sampler.run_mcmc(pos, n_steps, rstate0=state, progress=True)

#state = sampler.run_mcmc(init_params,n_steps, progress=True)

##########GET THE ACTUAL SAMPLES. Samples will have shape (n_walkers, n_params), eg. samples[:,0] is the density of the first parameter
samples = sampler.get_chain(flat=True)
# ndiscard = 50
# nthin    = 10
# sample   = sampler.get_chain(discard=ndiscard, 
#                              thin=nthin, 
#                              flat=True)
np.save('MCMC_samples.npy', samples)
# fig, ax = plt.subplots(2, 1, sharex=True)
# for i in [0, 1]:
#     ax[i].plot(sampler.chain[0,:,i], 'k-', lw=0.2)
#     ax[i].plot([0, n_steps-1], 
#              [sampler.chain[0,:,i].mean(), sampler.chain[0,:,i].mean()], 'r-')

# ax[1].set_xlabel('sample number')
# ax[0].set_ylabel('r')
# ax[1].set_ylabel('p')

#plt.show()
print('done')
print(samples)
quit()