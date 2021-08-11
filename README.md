Statistical analysis of PDFs and NNPDFs, focusing on uncertainty estimation. This uses xFitter 2.0.1 (https://www.xfitter.org/xFitter/), make sure you have LHAPDF and APPLgrid enabled with xFitter, with:

cd [path to xFitter];   ./configure --enable-lhapdf --enable-apfel --enable-applgrid  

If you don't have xFitter (or its dependencies such as QCDNUM, lapack, etc.) installed, first run 
cd install_xfitter && chmod +x install-lapack && ./install-lapack
(it is recommended that you make one directory, for example "external" and install everything on there, then export it as a global environment variable with 

export EXTERNAL=$HOME/external

then, add it to PATH and LD_LIBRARY_PATH, with

export PATH=$EXTERNAL/bin:$PATH && export LD_LIBRARY_PATH=$EXTERNAL/lib:LD_LIBRARY_PATH
)

then, in your installation directoy, run

chmod +x install-xfitter && ./install-xfitter 

Install any LHAPDF PDF set, with lhapdf command, or from https://lhapdf.hepforge.org/pdfsets (and unpack)
  
cd [path to LHAPDF sets]   ;    lhapdf --pdfdir=./ install CT14nlo
  
And add LHAPDF sets to $LHAPATH
  
export LHAPATH=[path to LHAPDF sets]/:$LHAPATH


To make quick plots of comparisons of different fits, do for example
  
xfitter-draw HERAPDF/HERA1_2_DIS/run/output:HERAPDF CTEQ14/HERA1_plus_2_combined/run/output:CTEQ14NLO --outdir HERA_CTEQ14_comparison --asym --bands --therr

