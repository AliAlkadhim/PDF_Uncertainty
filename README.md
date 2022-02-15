Thiss repository is intended for the statistical analysis of PDFs (especially NNPDF), focusing on uncertainty estimation. Note: you don't need to install anything other than the repository to analyze the data that is relevant for reweighting studies.

# xFitter Installation
This uses xFitter 2.0.1 (https://www.xfitter.org/xFitter/), as well as xfitter master (master is recommended). If you want to run the fits associated with xFitter, make sure you have LHAPDF and APPLgrid enabled with xFitter, with:

`cd [path to xFitter]`

`./configure ./configure --prefix=<installation path> --enable-lhapdf --enable-apfel --enable-applgrid ` 

If you don't have xFitter (or its dependencies such as QCDNUM, lapack, etc.) installed, first run 
`cd install_xfitter && chmod +x install-lapack && ./install-lapack`

(it is recommended that you make one directory, for example "external" and install everything on there, then export it as a global environment variable with 

`export EXTERNAL=$HOME/external`

then, add it to PATH and LD_LIBRARY_PATH, with

`export PATH=$EXTERNAL/bin:$PATH && export LD_LIBRARY_PATH=$EXTERNAL/lib:LD_LIBRARY_PATH`
)

then, in your installation directoy, run

`chmod +x install-xfitter && ./install-xfitter master`

# Installing PDF sets with lhapdf
Install any LHAPDF PDF set, with lhapdf command, or from https://lhapdf.hepforge.org/pdfsets (and unpack)
  
`cd [path to LHAPDF sets]`
`lhapdf --pdfdir=./ install CT14nlo`
  
And add LHAPDF sets to $LHAPATH
  
`export LHAPATH=[path to LHAPDF sets]/:$LHAPATH`

# xFitter

To make quick plots of comparisons of different fits, do for example
  
`xfitter-draw HERAPDF/HERA1_2_DIS/run/output:HERAPDF CTEQ14/HERA1_plus_2_combined/run/output:CTEQ14NLO --outdir HERA_CTEQ14_comparison --asym --bands --therr`


# Docker image for xFitter-master and this repository
xFitter has a huge number of dependencies (ROOT, Blas/lapack, QCDNUM, APFEL, FASTNLO, etc.) and xFitter-master has even more dependencies (such as yaml, cmake, etc.), and they all need to be installed locally, which could be a huge problem when a user is using a different architecture or environment. For this reason we have made an xFitter Docker image uses Ubuntu 20.04 with xFitter and its dependencies installed. This was done with the help of conda environments (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and ROOT on conda-forge (https://anaconda.org/conda-forge/root/). 

