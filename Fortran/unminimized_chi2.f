        subroutine unminimized_chi2
        implicit none
         
#include "fcn.inc"
#include "endmini.inc"
c note that these include files should also be included in main.f, since we're calling this from main.f
 
      integer A_MNE          !> Number of external parameters
c A for actual, as opposed to the max 200
c      parameter(MNE=200)

c         integer npar,iflag
c      double precision parminuitsave(MNE)
       integer i
       integer r, c
       integer r_max
      parameter(r_max=100)
      real MVN_cp(r_max, 2)
c       Double precision MVN_params

       Double precision chi2_
       integer iflag
c iflag from fnc.inc
        
       double precision chi2data_theory
c the type for the function that calculates it
       double precision chi2_per_dof
c c COMMON BLOCKS : if you #include #commonblock.inc you dont have to do /common...
c             common/endmini/pkeep,parminuitsave
c             Common/CFCN/IFlagFCN,nparFCN,ndfMINI,IfcnCount

c ------------------------------this block works
      iflag = IFlagFCN
c IFlagFCN from fcn.inc
      open(929, file='pars_saved.txt')
      chi2_ = chi2data_theory(iflag)
c calculate the chi2 at the iflag
      chi2_per_dof = chi2_/ndfMINI

      A_MNE = 198
      do i=1, A_MNE 
c      parminuit_cp(i) = parminuitsave(i)
c      parminuitsave(i) = parminuit(i)
      if (parminuitsave(i) .NE. 0.0) then
            write(929,668) parminuitsave(i)
c this to get rid of the 0 param values at the indeces. Otherwise use the function to get the parameter indeces, and print the paraminuitsave at those indeces
668   format(F10.8)
      endif
      enddo

      write(929,669) chi2_, chi2_per_dof
669   format(' ', D10.5, D10.5)     
      close(929)  
c --------------------------------------------

c access the MVN common block and write out the values to a file
      call read_MVN
      MVN_cp = MVN_params 

      open(930, file='MVN_saved.txt') 

      do r=1, r_max
            do c=1, 2
                  write(*, *) MVN_cp(r, c)
931   format((100F10.4,2F10.4))
            enddo
      enddo    

      close(930)

c print *, 'HIIIII \n \n HIIIIIIIIIIIIIIIIIII'
c print *, 'The unminimized chi2 value = ', chi2_

C Store params in a common block:
c if (iflag.eq.3) then
c do i=1,MNE
c    parminuitsave(i) = parminuit(i)
c    print *, 'the ith parameter is =', parminuit(i), '\n'
c    print *, 'the chi2 value is = \n', chi2out 
c enddo
c endif
        end



        subroutine read_MVN
        implicit none
#include "endmini.inc"
c
c This program reads n points from a data file and stores them in
c 3 arrays x, y, z.
c it then stores it in an array MVN_params(i,j) that is 100000X16 in common block common/MVN 
c
        integer nmax
        integer i, n, j
        parameter (nmax=100000)
              real MVN_cp(nmax, 14)

c these are the MVN samples of HERAPDF parameters
c these should be real since they were produced as floats
        real Bg(nmax), Cg(nmax), Aprig(nmax), 
     $       Bprig(nmax), Buv(nmax), Cuv(nmax), 
     $       Euv(nmax), Bdv(nmax), Cdv(nmax), 
     $       CUbar(nmax), DUbar(nmax), ADbar(nmax), 
     $       BDbar(nmax), CDbar(nmax)
             integer r, c
c Open the data file
        open (unit=20, file='/home/ali/Desktop/Research/xfitter/xfitter-2.0.1/datafiles/MVN.dat')

c the max number of sample is the number of rows from the common bloacl
c Read the number of points
        read(20,*) n
c        if (n.GT.nmax) then
        if (nmax.GT.n) then

        write(*,*) 'Error: n = ', n, 'is larger than nmax =', nmax
        goto 9999
        endif
c Loop over the data points: n is the total number of samples, which is the header in the file
c nmax is the number of samples we want to read is n
        do i=1, nmax
        read(20,100) Bg(i), Cg(i), Aprig(i), 
     $   Bprig(i), Buv(i), Cuv(i), Euv(i), 
     $   Bdv(i), Cdv(i), CUbar(i), DUbar(i), 
     $   ADbar(i), BDbar(i), CDbar(i)
c         write(*,*) Bg(i), Cg(i), Aprig(i), 
c      $   Bprig(i), Buv(i), Cuv(i), Euv(i), 
c      $   Bdv(i), Cdv(i), CUbar(i), DUbar(i), 
c      $   ADbar(i), BDbar(i), CDbar(i)
         enddo
100     format (14(F10.4))

c store the values of the parameter arrays in a multidimensional array
      do i=1, nmax
            MVN_cp(i, 1) = Bg(i)
            MVN_cp(i, 2) = Cg(i)
            MVN_cp(i, 3) = Aprig(i)
            MVN_cp(i, 4) = Bprig(i)
            MVN_cp(i, 5) = Buv(i)
            MVN_cp(i, 6) = Cuv(i)
            MVN_cp(i, 7) = Euv(i)
            MVN_cp(i, 8) = Bdv(i)
            MVN_cp(i, 9) = Cdv(i)
            MVN_cp(i, 10) = CUbar(i)
            MVN_cp(i, 11) = DUbar(i)
            MVN_cp(i, 12) = ADbar(i)
            MVN_cp(i, 13) = BDbar(i)
            MVN_cp(i, 14) = CDbar(i)

      enddo

      do i=1, nmax
            do j=1, 14
                  MVN_params(i,j) = MVN_cp(i,j)
            enddo
      enddo
c i is rows, j is cols   

c      open(909, file='MVN_pars.dat')
c      do 901 i = 1, n
c            Bg(i) = MVN_params(i,1) 
c            Cg(i) = MVN_params(i,2)
c       read(20, 202) (MVN_cp(j, j), j=1, 14)
c 901   continue      
c 202   format(F10.5)
c c      enddo 
c       do 902 i = 1, nmax-1
c       write(909, *) (MVN_cp(i, j), j=1, 14)
c 902   continue
c       close(909)
c       close(20)
      

c       do i=1, nmax
c             do j=1, 2
c                   read(20,100) MVN_cp(i,j)
c c                  write(*,*)  MVN_cp(i,j)
c                   MVN_params(i,j) = MVN_cp(i,j)
c             enddo
c       enddo

c 100     format (2(F10.4))




c Close the file
        close (20)
c save the MVN common block array into common blocl MVN_params
 
c        do r=1, nmax
c                do c=1, 2
c                        MVN_params(1,r) = Bg(r)
c                enddo
c        enddo
c Now we can process the data somehow...
9999    stop
        end 
