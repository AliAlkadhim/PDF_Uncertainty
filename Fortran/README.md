MVN.dat is the data file containing the multivariate gaussian sampling, from the best-fit values and covariance matrix.
To see how this data is generated, see for example output_example_run/minuit_out_processing.ipynb
read_MVN.f reads the data from MVN.dat into the 14 arrays: one for each of the HERAPDF parameters.

TODO
Next work is in main, to read each of the vectors of parameters and calculate the chi2 for it....
