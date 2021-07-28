        subroutine unminimized_chi2
         
#include "fcn.inc"
#include "endmini.inc"

 
c        external fcn

        integer npar,iflag
      double precision parminuit(MNE),chi2out
      integer i
      double precision chi2data_theory
c COMMON BLOCKS : I think if you #include #commonblock.inc you dont have to do /common...
            common/endmini/pkeep,parminuitsave
            Common/CFCN/IFlagFCN,nparFCN,ndfMINI,IfcnCount

      IFlagFCN = IFlag

      NparFCN  = npar

c    call chi2data_theory(iflag)
      print *, 'HIIIII \n \n HIIIIIIIIIIIIIIIIIII'
      print *, 'The unminimized chi2 value = ', chi2out

C Store params in a common block:
        chi2out = chi2data_theory(iflag)
      if (iflag.eq.3) then
      do i=1,MNE
         parminuitsave(i) = parminuit(i)
         print *, 'the ith parameter is =', parminuit(i), '\n'
         print *, 'the chi2 value is = \n', chi2out 
      enddo
      endif
        end