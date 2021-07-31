        subroutine read_MVN
#include "endimi.inc"
c
c This program reads n points from a data file and stores them in
c 3 arrays x, y, z.
c it then stores it in an array MVN_params(i,j) that is 100000X16 in common block common/MVN 
c
        integer nmax
        parameter (nmax=100000)
c these are the HERAPDF parameters
        real Bg(nmax), Cg(nmax), Aprig(nmax), 
     $       Bprig(nmax), Buv(nmax), Cuv(nmax), 
     $       Euv(nmax), Bdv(nmax), Cdv(nmax), 
     $       CUbar(nmax), DUbar(nmax), ADbar(nmax), 
     $       BDbar(nmax), CDbar(nmax)
c Open the data file
        open (unit=20, file='MVN.dat')
c Read the number of points
        read(20,*) n
        if (n.GT.nmax) then
        write(*,*) 'Error: n = ', n, 'is larger than nmax =', nmax
        goto 9999
        endif
c Loop over the data points
        do i= 1, nmax
        read(20,100) Bg(i), Cg(i), Aprig(i), 
     $   Bprig(i), Buv(i), Cuv(i), Euv(i), 
     $   Bdv(i), Cdv(i), CUbar(i), DUbar(i), 
     $   ADbar(i), BDbar(i), CDbar(i)
        write(*,*) Bg(i), Cg(i), Aprig(i), 
     $   Bprig(i), Buv(i), Cuv(i), Euv(i), 
     $   Bdv(i), Cdv(i), CUbar(i), DUbar(i), 
     $   ADbar(i), BDbar(i), CDbar(i)
        enddo
100     format (14(F10.4))

c Close the file
        close (20)
c save the MVN common block array into common blocl MVN_params
        integer r, c
        do r=1, 100000
                do c=1, 14
                        MVN_params(r,0) = Bg(r)
                enddo
        enddo
c Now we can process the data somehow...
9999    stop
        end 