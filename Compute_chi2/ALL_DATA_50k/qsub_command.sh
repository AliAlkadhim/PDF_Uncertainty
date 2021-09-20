#!/bin/sh

ls -la /cvmfs/cms.cern.ch/
source /cvmfs/cms.cern.ch/cmsset_default.sh

source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/setup_xfitter.sh
cd /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/RUNS/ALL_DATA_50k

###python /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/compute_chi2.py
### always run in cwd, if you forget to do 'qsub -cwd'
####$ -cwd

for ((i = 0 ; i < 5 ; i++)); do
    cd job_$i
    qsub run_job.sh
    cd ..
done
