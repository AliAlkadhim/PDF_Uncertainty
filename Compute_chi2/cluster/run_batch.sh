#!/bin/sh
###set the qsub options are set in lines that begin with #$

### -S pathname: Specifies the interpreting shell for the job
#$ -S /bin/sh

### -o pathname: The path used for the standard  output  stream  of  the job. the -j y specifies that the standard error (logs) should be merged with the standard out of the job.
#$ -o /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/logs -j y
start=`date +%s`

###PBS: unsure if we have this for now.
###PBS -l nodes=12:ppn=12
source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/setup_xfitter.sh
cd src/RUNS/example_batch_job
###cd $PWD
### Job name
###PBS -N ExampleJob

### Number of nodes
###PBS -l nodes=4
###PBS -l nodes=4:compute#shared

###echo PBS default server is $PBS_DEFAULT

####number of slots
###$ NSLOTS=10
###6 parallel processes
###PARALLEL ENV: smp, orte do not exis
###$ -pe mpi 6- 

###python /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/compute_chi2.py
### always run in cwd, if you forget to do 'qsub -cwd'
#$ -cwd

python compute_chi2.py


end=`date +%s`
runtime=$((end-start))
echo $runtime >> time.txt
