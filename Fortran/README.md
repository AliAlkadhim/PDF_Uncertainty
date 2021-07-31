MVN.dat is the data file containing the multivariate gaussian sampling, from the best-fit values and covariance matrix.

read_MVN.f reads the data from MVN.dat into the 14 arrays: one for each of the HERAPDF parameters. Then, subroutine read_MVN in unminimized_chi2.f reads this data and stores it in a 2-dimensional array (MVN_params) of size N_{samples} X N_{parameters} in the common block endmini, to be used by other pars of the program. Each row of this array can be regarded as a different set of parameter values. 


TODO: We wish to modify "fcn.f" and "main.f", so that we use all the functionality of xfitter, except the minimization parts. "temp_fcn.f" is a temporary version of xfitter's fcn.f, since we need to modify it for our purpose. Also, "call unminimized_chi2" must be inserted in "main.f" (this works). We wish to calculate the chi2 value, using the subroutine "chi2data_theory" which is in "fcn.f" between the central parameter set (which is defined in "minuit.in.txt") and the sampled parameter set (from MVN_pars), without doing any minimization (without calling minuit). This is tricky since all the parts of xfitter depend on the minimization, and the chi2data_theory takes iflag as input parameter, which is the flag that is set by minuit. An instance of parameter values should be copied to parminuitsave which comes from pkeep which is set at each point of the minimization by minuit...

There is also minimal notes in TODO.

(To see how this data is generated, see for example output_example_run/minuit_out_processing.ipynb)
