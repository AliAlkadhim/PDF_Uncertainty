        subroutine unminimized_chi2
        implicit none
         
#include "fcn.inc"
#include "endmini.inc"
c note that these include files should also be included in main.f, since we're calling this from main.f
 
      integer A_MNE          !> Number of external parameters
c      parameter(MNE=200)

c         integer npar,iflag
c      double precision parminuitsave(MNE)
       integer i
       Double precision chi2_
       integer iflag
        
       double precision chi2data_theory
       double precision chi2_per_dof
c c COMMON BLOCKS : if you #include #commonblock.inc you dont have to do /common...
c             common/endmini/pkeep,parminuitsave
c             Common/CFCN/IFlagFCN,nparFCN,ndfMINI,IfcnCount

      iflag = IFlagFCN
      open(929, file='pars_saved.txt')
      chi2_ = chi2data_theory(iflag)
      chi2_per_dof = chi2_/ndfMINI

      A_MNE = 15
      do i=1, A_MNE 
c      parminuit_cp(i) = parminuitsave(i)
c      parminuitsave(i) = parminuit(i)
      write(929,668) parminuitsave(i)
668   format(F10.8)
      enddo

      write(929,669) chi2_, chi2_per_dof
669   format(' ', D10.5, D10.5)     
      close(929)          
      print *, 'HIIIII \n \n HIIIIIIIIIIIIIIIIIII'
      print *, 'The unminimized chi2 value = ', chi2_

C Store params in a common block:
c if (iflag.eq.3) then
c do i=1,MNE
c    parminuitsave(i) = parminuit(i)
c    print *, 'the ith parameter is =', parminuit(i), '\n'
c    print *, 'the chi2 value is = \n', chi2out 
c enddo
c endif
        end
