#!/bin/sh
###set the qsub options are set in lines that begin with #$

### -S pathname: Specifies the interpreting shell for the job
#$ -S /bin/sh

### -o pathname: The path used for the standard  output  stream  of  the job. the -j y specifies that the standard error (logs) should be merged with the standard out of the job.
#$ -o /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/logs -j y
start=`date +%s`

#$ -v SGE_STDOUT_PATH
###PBS: unsure if we have this for now.
###PBS -l nodes=12:ppn=12
### note that setup_xfitter.sh changes directory to $CMSSW_BASE
source /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/setup_xfitter.sh
cd src/RUNS/example_batch_job
###cd $PWD
### say we want to run qsub for each 10 parameter set samples, this means we have to create a diirectory for each sample

####number of slots
###$ NSLOTS=10
###6 parallel processes
###PARALLEL ENV: smp, orte do not exis
###$ -pe mpi 6- 

###python /storage/5/home/aalkadhim/xfitter/CMSSW_10_2_25/src/RUNS/example_batch_job/compute_chi2.py
### always run in cwd, if you forget to do 'qsub -cwd'
#$ -cwd

for ((i = 0 ; i < 10 ; i++)); do
   dir_name = job_$i
	mkdir dir_name
	cd dir_name
	cp ../steerings.txt .
	cp ../ewparam.txt .
	cp ../compute_chi2.py .
	sed 's/dir/dir_num' compute_chi2.py
### the & tells linux to move on to the next line without having to wait for this line's command to finish. 
	python compute_chi2.py &
	

done

python compute_chi2.py


end=`date +%s`
runtime=$((end-start))
echo $runtime >> time.txt
