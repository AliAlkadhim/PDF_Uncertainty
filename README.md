Statistical analysis of PDFs and NNPDFs, focusing on uncertainty estimation. This uses xFitter, make sure you have LHAPDF and APLEF enabled with xFitter, with:
cd [path to xFitter];   ./configure --enable-lhapdf --enable-nnpdf --enable

Install any LHAPDF PDF set, with lhapdf command, or from https://lhapdf.hepforge.org/pdfsets (and unpack)
  
cd [path to LHAPDF sets]   ;    lhapdf --pdfdir=./ install CT14nlo
  
And add LHAPDF sets to $LHAPATH
  
export LHAPATH=[path to LHAPDF sets]/:$LHAPATH


To make quick plots of comparisons of different fits, do for example
  
xfitter-draw HERAPDF/HERA1_2_DIS/run/output:HERAPDF CTEQ14/HERA1_plus_2_combined/run/output:CTEQ14NLO --outdir HERA_CTEQ14_comparison --asym --bands --therr

