export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/numdiff/bin/:$PATH
export CURRENTDIR=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src
export version=`cat version`
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/xfitter-master/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/hoppet/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/applgrid/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/lhapdf/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/apfel/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/mela/bin:$PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/apfelxx/bin:$PATH
export PYTHONPATH=:/cvmfs/cms.cern.ch/slc6_amd64_gcc700/external/lhapdf/6.2.1-gnimlf4/lib/python2.7/site-packages:/cvmfs/cms.cern.ch/slc6_amd64_gcc700/external/lhapdf/6.2.1-gnimlf4/lib/python2.7/site-packages 
export LD_LIBRARY_PATH=$CURRENTDIR/deps/hoppet/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENTDIR/deps/lhapdf/lib/:$LD_LIBRARY_PATH
export HATHOR_DIR=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/Hathor-2.0
export LD_LIBRARY_PATH=$CURRENTDIR/deps/applgrid/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENTDIR/deps/apfel/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENTDIR/deps/mela/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENTDIR/deps/qcdnum/lib/:$LD_LIBRARY_PATH
export PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/qcdnum/bin:$PATH
chmod +x /storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps/qcdnum-17-01-14/bin/qcdnum-config
. /cvmfs/sft.cern.ch/lcg/contrib/gcc/7/x86_64-slc6/setup.sh
cd /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/5.34.36/x86_64-slc6-gcc7-opt/root/
. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/5.34.36/x86_64-slc6-gcc7-opt/root/bin/thisroot.sh 
cd -
source /storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/setup_yaml.sh
cd $CURRENTDIR
