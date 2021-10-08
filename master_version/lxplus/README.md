Directions for installation in lxplus:

$export SCRAM_ARCH=slc7_amd64_gcc900
$ cmsrel CMSSW_12_0_0_pre4

$ cd CMSSW_12_0_0_pre4/src
$ cmsenv
$  git clone https://gitlab.cern.ch/fitters/xfitter.git

$  cd xfitter/tools/

$./install-lapack
$./setup_lapacl.sh

$./install-yaml

$./setup_yaml.sh

$ cd ../

$./make.sh install
