#!/bin/sh
###set the qsub options are set in lines that begin with #$
### -S pathname: Specifies the interpreting shell for the job

#$ -S /bin/sh

### -o pathname: The path used for the standard  output  stream  of  the job. the -j y specifies that the standard error (logs) should be merged with the standard out of the job.
#$ -o /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/RUNS/logs -j y

p=$(pwd)
source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/setup_xfitter.sh
cd /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_27/src/RUNS/ALL_DATA_50k
###export PATH=$PATH:/storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/xfitter-2.0.1/src
cd $p




#$ -cwd

python compute_chi2.py
