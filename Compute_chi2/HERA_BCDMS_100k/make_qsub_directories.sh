#!/bin/sh


### note that setup_xfitter.sh changes directory to $CMSSW_BASE
###source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/setup_xfitter.sh
cd /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/RUNS/ALL_DATA_50k
### say we want to run qsub for each 10 parameter set samples, this means we have to create a diirectory for each sample


for ((i = 0 ; i < 5 ; i++)); do
 ###  dir_name = job_$i
### run directory index is the same as the "sample_ind" in the python file (a directory for each sample)
	mkdir -p job_$i
	cd job_$i
	cp ../steering.txt .
	cp ../ewparam.txt .
	cp ../compute_chi2.py .
	cp ../minuit.in.txt .
	ln -s /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/xfitter-2.0.1/datafiles datafiles
	old=sample_ind
	sed -i "s|$old|$i|g" compute_chi2.py
    cp ../run_job.sh .

	cd ..

done


