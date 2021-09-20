#!/bin/sh

###source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/setup_xfitter.sh
###cd src/RUNS/example_batch_job

###python /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/compute_chi2.py
### always run in cwd, if you forget to do 'qsub -cwd'
####$ -cwd

for ((i = 0 ; i < 10 ; i++)); do
    cd job_$i
    qsub run_job.sh
    cd ..
done
