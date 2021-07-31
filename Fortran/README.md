MVN.dat is the data file containing the multivariate gaussian sampling, from the best-fit values and covariance matrix.
To see how this data is generated, see for example output_example_run/minuit_out_processing.ipynb
read_MVN.f reads the data from MVN.dat into the 14 arrays: one for each of the HERAPDF parameters.

Then, subroutine read_MVN in unminimized_chi2.f reads this data and stores it in a 2-dimensional array (MVN_params) of size N_{samples} X N_{parameters} in the common block endmini, to be used by other pars of the program. Each row of this array can be regarded as a different set of parameter values. 
