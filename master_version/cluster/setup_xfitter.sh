#!/bin/sh
ls -la /cvmfs/cms.cern.ch/
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src
cmsenv

####LHAPDF
export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/LHAPDF-6.2.3/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/LHAPDF-6.2.3/src

export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/qcdnum-17-01-15/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/qcdnum-17-01-15/lib

###ZLIB
export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/zlib-1.2.11
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/zlib-1.2.11

export C_INCLUDE_PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/zlib-1.2.11
export CPLUS_INCLUDE_PATH=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/zlib-1.2.11
##########

####APPLGRID
export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/applgrid-1.6.17/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/applgrid-1.6.17/src/.libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/applgrid-1.6.17/src

#########XFITTER
export PATH=$PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/xfitter-2.0.1/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/xfitter-2.0.1/tools
