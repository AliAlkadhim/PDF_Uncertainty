#!/bin/bash
#####################################################################
## Configuration ####################################################

## Programs versions
lhapdfver=6.2.1
hathorver=2.0
hoppetver=1.2.0
applgridver=1.5.34
qcdnumver=17-01-14
apfelver=3.0.4
melaver=2.0.1
apfelgridver=1.0.1
apfelxxver=4.0.0

# skip some packages depending on xfitter version
skip_mela=0
skip_apfelgrid=0

## Optional manual configurations
#MANUALCONF=1

## Option 1: Use local environment
## These settings assume that root is already in your PATH and boost lib is installed
## For versions older than 1.1.0, need to specify the location of CERNLIB
#MODE=local
#export CERN_ROOT=/usr/lib

## Option 2: Setup environment from CERN cvmfs
## These settings include compiler, cernlib and root
MODE=cern
gccv=4.6
os=slc6
arch=x86_64
rootversion=5.34.18

## End of Configuration ####################################################
#####################################################################
#Check for dummies
if [[ $0 == bash || $0 = csh ]]
then
    echo "Please don't source me, I am an executable!"
    echo "Give me attributes with:"
    echo "chmod +x install-xfitter"
    echo "and run me with:"
    echo "./install-xfitter"
    return 2
fi

if [[ -z $1 ]]
then
    echo
    echo "usage:"
    echo "$0 <version|deps>"
    echo
    echo "available versions:"
    vers=`git ls-remote --tags https://gitlab.cern.ch/fitters/xfitter.git | sed 's|/| |g; s|\^| |' | awk '{print $4}' | uniq`
    echo "$vers"
    echo "master"
    echo
    echo "to reinstall only dependences, run:"
    echo "$0 deps"
    echo
    exit
fi
mode=$1
shift 1

#in deps mode, read version from the version file
if [[ $mode != "deps" ]]
    then
    version=$mode
else
    if [[ ! -e version ]]
        then
        # assume it is master (e.g. for gitlab CI)
        version="master"
        #
        #echo
        #echo "could not find file \"version\""
        #echo "cannot determine current version"
        #echo "run first:"
        #echo "$0 <version>"
        #echo
        #exit
    else
        version=`cat version`
    fi
    echo "reinstalling dependencies for xFitter version $version"
fi
# strip the xfitter version
stripversion=`echo $version |sed "s/\.//g"`


#check that requested version exists
if [[ $version == "2.0.0" ]] || [[ $version == "2.0.1" ]]
    then
    # use older QCDNUM for xfitter-2.0.0
    if [[ $version == "2.0.0" ]]; then
      qcdnumver=17-01-13
    fi
    exist=0
    for ver in ``
    do
        if [[ $version == $ver ]]
        then
            exist=1
        fi
    done

    vers=`git ls-remote --tags https://gitlab.cern.ch/fitters/xfitter.git | sed 's|/| |g; s|\^| |' | awk '{print $4}' | uniq`

    for ver in $vers
    do
        if [[ $version == $ver ]]
        then
            exist=1
        fi
    done

    if [[ $exist == 0 ]]
    then
        echo
        echo "version $version not found, available versions:"
        echo ""
        echo "$vers"
        echo "master"
        echo
        exit
    fi
fi

if [[ $mode != "deps" && -e xfitter-${version} &&  -e herafitter-${version} ]]
then
    echo
    echo "xfitter-${version} already exists, remove it first"
    echo "To reinstall only dependences, run:"
    echo "$0 deps"
    echo
    exit
fi
#if [[ $mode == "deps" && ! -e xfitter-${version}  && ! -e herafitter-${version}  ]]
#then
#    echo
#    echo "xfitter-${version} or herafitter-${version}   does not exist, install it first with:"
#    echo "$0 $version"
#    echo
#    exit
#fi

# skip some packages depending on xfitter version
if [[ $version == "master" ]]; then
  skip_mela=1
  skip_apfelgrid=1
fi

#automatically detect system:
if [[ -z $MANUALCONF ]]
then
    which sw_vers >& /dev/null
    if [[ $? == 0 ]]
    then
        echo "Detected Mac OS X system"
        MODE=local
    else
        SYS=$(echo `lsb_release -i |cut -d: -f2`)
        ver=$(echo `lsb_release -r |cut -d: -f2`)
        if [[ $SYS == Scientific* && $ver == 6.* ]]
        then
            echo "Detected SL6 Linux distribution"
            MODE=cern
            gccv=4.9
            echo "Using gcc version = ${gccv}"
            os=slc6
            arch=x86_64
            rootversion=5.34.36
            boostver=1.53.0
            pyth=2.7
        elif [[ $SYS == CentOS* && $ver == 7.* ]]
        then
            echo "Detected CentOS7 Linux distribution"
            MODE=cern
            gccv=4.9
            os=centos7
            arch=x86_64
            rootversion=6.06.08
            boostver=1.53.0
            pyth=2.7
        elif [[ $SYS == Scientific* && $ver == 5.* ]]
        then
            echo "Detected SL5 Linux distribution"
            MODE=cern
            gccv=4.3
            os=slc5
            arch=x86_64
            rootversion=5.34.00
            boostver=1.48.0
            python=2.6.5
            pyth=2.6
        elif [[ $SYS == "Ubuntu" ]]
        then
            echo "Detected Ubuntu distribution"
            MODE=local
        else
            echo "Sorry, I don't recognize your system:"
            echo "$SYS $ver"
            echo "I will assume you have root installed in your system,"
            echo "gcc version >= 4.3, python, boost libraries, and wget"
            echo "If this doesn't work, and you have /cvmfs/sft.cern.ch mounted"
            echo "edit me (I am $0) and try to setup appropriate settings"
            echo "in the section: manual configuration"
            echo
            MODE="local"
        fi
    fi
fi
if [[ $MODE == "cern" ]]
    then
#    if [[ ! -e /afs/cern.ch ]]
    if [[ ! -e /cvmfs/sft.cern.ch ]]
        then
        echo
        echo "/cvmfs/sft.cern.ch not mounted, forcing local MODE"
        echo "Fasten you seat belt"
        echo "I hope you have root, gcc >= 4.8, python and boost libraries"
        echo "all installed in your system"
        echo
        MODE="local"
    fi
fi

if [[ $MODE == "cern" ]]
then
    compiler=`echo gcc${gccv} | sed "s/\.//"`
#    . /afs/cern.ch/sw/lcg/contrib/gcc/${gccv}/${arch}-${os}/setup.sh
    . /cvmfs/sft.cern.ch/lcg/contrib/gcc/${gccv}/${arch}-${os}/setup.sh
#    . /afs/cern.ch/sw/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root/bin/thisroot.sh
    . /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root/bin/thisroot.sh
    if [[ $os == slc5 ]]
    then
        echo "LEGACY SL5 ! using afs"
        PYTHONBIN=/afs/cern.ch/sw/lcg/external/Python/${python}/${arch}-${os}-${compiler}-opt/bin
        PATH=$PYTHONBIN:$PATH
        export BOOST=--with-boost=/afs/cern.ch/sw/lcg/external/Boost/${boostver}_python${pyth}/${arch}-${os}-${compiler}-opt
    fi
    if [[ $os == slc6 ]]
    then
        export BOOST=--with-boost=/cvmfs/sft.cern.ch/lcg/external/Boost/${boostver}_python${pyth}/${arch}-${os}-${compiler}-opt
    fi
fi

#check some basic dependendencies before starting the installation
which git >& /dev/null
if [[ $? != 0 ]]
then
    echo "Error, git not found"
    exit
fi

which root >& /dev/null
if [[ $? != 0 ]]
then
    echo "Error, root not found"
    exit
fi

which wget >& /dev/null
if [[ $? == 0 ]]
then
    http=wget
else
    which curl >& /dev/null
    if [[ $? == 0 ]]
    then
        http=curl
    else
        echo "Error, wget or curl not found"
        exit
    fi
fi

#directory:
CURRENTDIR=`pwd`

#clean up
rm version setup.sh compile quickstart.readme.txt >& /dev/null
rm install.log >& /dev/null



# keep for debugging
installDeps=1


if [[ $installDeps == 0 ]]
then
   echo "Skip installation of dependences"
else
#Make all dependencies
    rm -rf deps >& /dev/null
    mkdir deps
    cd deps
#lhapdf:
    echo "Installing LHAPDF $lhapdfver..."
    if (( `echo $lhapdfver |cut -d. -f1` >= 6 ))
    then
        lhapdf="LHAPDF"
        withboost=$BOOST
    else
        lhapdf="lhapdf"
    fi

    if [[ $http == "curl" ]]
    then
#       curl https://www.hepforge.org/archive/lhapdf/${lhapdf}-${lhapdfver}.tar.gz > ${lhapdf}-${lhapdfver}.tar.gz 2>> $CURRENTDIR/install.log
        curl https://lhapdf.hepforge.org/downloads/${lhapdf}-${lhapdfver}.tar.gz > ${lhapdf}-${lhapdfver}.tar.gz 2>> $CURRENTDIR/install.log
    else
#       wget https://www.hepforge.org/archive/lhapdf/${lhapdf}-${lhapdfver}.tar.gz >> $CURRENTDIR/install.log 2>&1
        wget https://lhapdf.hepforge.org/downloads/${lhapdf}-${lhapdfver}.tar.gz >> $CURRENTDIR/install.log 2>&1
    fi
    tar xfz ${lhapdf}-${lhapdfver}.tar.gz  >> $CURRENTDIR/install.log 2>&1
    cd ${lhapdf}-${lhapdfver}
    ./configure --prefix=$CURRENTDIR/deps/lhapdf  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make -j 9  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    cd - >& /dev/null

#Hathor:
    # echo "Installing Hathor $hathorver..."
    # hathor="Hathor"
    # if [[ $http == "curl" ]]
    # then
    #     # alternative link https://www-zeuthen.desy.de/~moch/hathor/Hathor-2.0.tar.gz
    #     curl https://www.physik.hu-berlin.de/de/pep/tools/Hathor-${hathorver}.tar.gz 2>> $CURRENTDIR/install.log
    # else
    #     wget https://www.physik.hu-berlin.de/de/pep/tools/Hathor-${hathorver}.tar.gz >> $CURRENTDIR/install.log 2>&1
    # fi
    # tar xfz Hathor-${hathorver}.tar.gz  >> $CURRENTDIR/install.log 2>&1
    # cd Hathor-${hathorver}/lib
    # # need to provide LHAPDF directory and add -fPIC flag to CFLAGS and FFLAGS
    # make LHAPDF=$CURRENTDIR/deps/lhapdf V=1 CFLAGS='-O2 -Wall -fPIC' FFLAGS='-ffixed-line-length-132 -fPIC' -j9 >> $CURRENTDIR/install.log  2>&1
    # if [[ $? != 0 ]]
    # then
    #     echo "Error, check install.log for details"
    #     exit
    # fi
    # cd - >& /dev/null

 #hoppet:
    echo "Installing HOPPET $hoppetver..."
    if [[ $http == "curl" ]]
    then
        curl http://hoppet.hepforge.org/downloads/hoppet-${hoppetver}.tgz > hoppet-${hoppetver}.tgz 2>> $CURRENTDIR/install.log
    else
        wget http://hoppet.hepforge.org/downloads/hoppet-${hoppetver}.tgz >> $CURRENTDIR/install.log 2>&1
    fi
    tar xfz hoppet-${hoppetver}.tgz  >> $CURRENTDIR/install.log  2>&1
    cd hoppet-${hoppetver}
    ./configure --prefix=$CURRENTDIR/deps/hoppet  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make -j 9  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
#       exit
    fi
    cd - >& /dev/null

 # setup paths for applgrid:
    export PATH=$CURRENTDIR/deps/hoppet/bin/:$PATH
    export PATH=$CURRENTDIR/deps/lhapdf/bin/:$PATH
    # export HATHOR_DIR=$CURRENTDIR/deps/Hathor-${hathorver}
    export LD_LIBRARY_PATH=$CURRENTDIR/deps/lhapdf/lib/:$LD_LIBRARY_PATH
    echo `lhapdf-config --prefix`/lib*/python*/site-packages > /tmp/xf_py1234_a
    export PYTHONPATH=$PYTHONPATH:`cat /tmp/xf_py1234_a`
    rm /tmp/xf_py1234_a

 #applgrid:
    echo "Installing APPLGRID $applgridver..."
    APPLGRID_URL=https://applgrid.hepforge.org/downloads/applgrid-"$applgridver".tgz
#   APPLGRID_URL=https://www.hepforge.org/archive/applgrid/applgrid-"$applgridver".tgz
    if [[ $http == "curl" ]]
    then
        curl $APPLGRID_URL > applgrid-$applgridver.tgz  2>> $CURRENTDIR/install.log
    else
        wget $APPLGRID_URL >> $CURRENTDIR/install.log 2>&1
    fi
    tar xfz applgrid-$applgridver.tgz  >> $CURRENTDIR/install.log  2>&1
    cd applgrid-$applgridver
    # need to supply c++11 flag explicitly
    ./configure CXXFLAGS='--std=c++11' --prefix=$CURRENTDIR/deps/applgrid  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make   >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make -j 9  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    cd - >& /dev/null

    export PATH=$CURRENTDIR/deps/applgrid/bin/:$PATH

 #apfel
    echo "Installing APFEL $apfelver..."
    if [[ $http == "curl" ]]
    then
        curl https://github.com/scarrazza/apfel/archive/${apfelver}.tar.gz > ${apfelver}.tar.gz 2 >> $CURRENTDIR/install.log
    else
        wget https://github.com/scarrazza/apfel/archive/${apfelver}.tar.gz >> $CURRENTDIR/install.log 2>&1
    fi
    mv ${apfelver}.tar.gz apfel-${apfelver}.tar.gz
    tar xfvz apfel-${apfelver}.tar.gz >> $CURRENTDIR/install.log 2>&1
    cd apfel-${apfelver}
    ./configure --prefix=$CURRENTDIR/deps/apfel --disable-lhapdf >> $CURRENTDIR/install.log  2>&1

    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make -j 9  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    cd - >& /dev/null
 # setup paths for apfel:
    export PATH=$CURRENTDIR/deps/apfel/bin/:$PATH

#Importing tables for Small-x resummation studies
    echo "Installing HELL tables..."
    if [[ $http == "curl" ]]
    then
      curl https://www.ge.infn.it/~bonvini/hell/downloads/HELLx.v3.0.tgz > HELLx.v3.0.tgz 2 >> $CURRENTDIR/install.log
      curl https://www.ge.infn.it/~bonvini/hell/downloads/HELLx-data.v3.tgz > HELLx-data.v3.tgz 2 >> $CURRENTDIR/install.log
    else
      wget https://www.ge.infn.it/~bonvini/hell/downloads/HELLx.v3.0.tgz >> $CURRENTDIR/install.log  2>&1
      wget https://www.ge.infn.it/~bonvini/hell/downloads/HELLx-data.v3.tgz >> $CURRENTDIR/install.log  2>&1
    fi

    tar xzf HELLx.v3.0.tgz >> $CURRENTDIR/install.log  2>&1
    tar xzf HELLx-data.v3.tgz >> $CURRENTDIR/install.log  2>&1

    cp -r HELLx/data/* $CURRENTDIR/deps/apfel/share/apfel/.

 #apfelgrid
    if [[ $skip_apfelgrid != 1 ]]; then
      if [ -d /cvmfs ]
      then
                  lhapdf get NNPDF30_nlo_as_0118  >> $CURRENTDIR/install.log  2>&1
          else
                  wget http://www.hepforge.org/archive/lhapdf/pdfsets/6.2/NNPDF30_nlo_as_0118.tar.gz  >> $CURRENTDIR/install.log  2>&1
                  tar xvzpf NNPDF30_nlo_as_0118.tar.gz >> $CURRENTDIR/install.log  2>&1
                  mv NNPDF30_nlo_as_0118 `lhapdf-config --datadir` >> $CURRENTDIR/install.log  2>&1
                  rm NNPDF30_nlo_as_0118.tar.gz  >> $CURRENTDIR/install.log  2>&1
          fi
          echo "Installing APFELgrid $apfelgridver..."
      # tmp solution is to use fork @zenaiev
      apfelgridver=1.0.5
      if [[ $http == "curl" ]]
      then
  #       curl https://github.com/nhartland/APFELgrid/archive/v${apfelgridver}.tar.gz > v${apfelgridver}.tar.gz 2 >> $CURRENTDIR/install.log
          curl https://github.com/zenaiev/APFELgrid/archive/v${apfelgridver}.tar.gz > v${apfelgridver}.tar.gz 2 >> $CURRENTDIR/install.log
      else
  #       wget https://github.com/nhartland/APFELgrid/archive/v${apfelgridver}.tar.gz >> $CURRENTDIR/install.log 2>&1
          wget https://github.com/zenaiev/APFELgrid/archive/v${apfelgridver}.tar.gz >> $CURRENTDIR/install.log 2>&1
      fi
      mv v${apfelgridver}.tar.gz APFELgrid-${apfelgridver}.tar.gz
      tar xfvz APFELgrid-${apfelgridver}.tar.gz >> $CURRENTDIR/install.log 2>&1
      cd APFELgrid-${apfelgridver}
      ./setup.sh  >> $CURRENTDIR/install.log  2>&1
      if [[ $? != 1 ]]
      then
          echo "Error, check install.log for details"
          exit
      fi
      cd - >& /dev/null
    fi
    
 #mela
    if [[ $skip_mela != 1 ]]; then
      echo "Installing MELA $melaver..."

      if [[ $http == "curl" ]]
      then
          curl https://github.com/vbertone/MELA/archive/${melaver}.tar.gz > ${melaver}.tar.gz 2 >> $CURRENTDIR/install.log
      else
          wget https://github.com/vbertone/MELA/archive/${melaver}.tar.gz >> $CURRENTDIR/install.log 2>&1
      fi
      mv ${melaver}.tar.gz MELA-${melaver}.tar.gz
      tar xfvz MELA-${melaver}.tar.gz >> $CURRENTDIR/install.log 2>&1
      cd MELA-${melaver}
      ./configure --prefix=$CURRENTDIR/deps/mela  >> $CURRENTDIR/install.log  2>&1

      if [[ $? != 0 ]]
      then
          echo "Error, check install.log for details"
          exit
      fi
      make -j 9  >> $CURRENTDIR/install.log  2>&1
      if [[ $? != 0 ]]
      then
          echo "Error, check install.log for details"
          exit
      fi
      cd - >& /dev/null
   # setup paths for mela:
      export PATH=$CURRENTDIR/deps/mela/bin/:$PATH
    fi
    
 #apfel
    echo "Installing APFELxx $apfelxxver..."
    if [[ $http == "curl" ]]
    then
        curl https://github.com/vbertone/apfelxx/archive/v${apfelxxver}.tar.gz > v${apfelxxver}.tar.gz 2 >> $CURRENTDIR/install.log
    else
        wget https://github.com/vbertone/apfelxx/archive/v${apfelxxver}.tar.gz  >> $CURRENTDIR/install.log 2>&1
    fi
    mv v${apfelxxver}.tar.gz apfelxx-${apfelxxver}.tar.gz
    tar xfvz apfelxx-${apfelxxver}.tar.gz >> $CURRENTDIR/install.log 2>&1

    cd apfelxx-${apfelxxver}
    cmake -DCMAKE_INSTALL_PREFIX=$CURRENTDIR/deps/apfelxx >> $CURRENTDIR/install.log  2>&1
    make >> $CURRENTDIR/install.log  2>&1
    make install >> $CURRENTDIR/install.log  2>&1
    
    export PATH=$CURRENTDIR/deps/apfelxx/bin/:$PATH
    cd - >& /dev/null
 #qcdnum
    echo "Installing QCDNUM $qcdnumver..."
    qcdnumstripver=`echo $qcdnumver |sed "s/-//g"`
    if [[ $http == "curl" ]]
    then
        curl http://www.nikhef.nl/user/h24/qcdnum-files/download/qcdnum${qcdnumstripver}.tar.gz > qcdnum${qcdnumstripver}.tar.gz 2>> $CURRENTDIR/install.log
    else
        wget http://www.nikhef.nl/user/h24/qcdnum-files/download/qcdnum${qcdnumstripver}.tar.gz >> $CURRENTDIR/install.log 2>&1
    fi
    tar xfz qcdnum${qcdnumstripver}.tar.gz  >> $CURRENTDIR/install.log  2>&1
    cd qcdnum-${qcdnumver}

    ./configure --prefix=$CURRENTDIR/deps/qcdnum  >> $CURRENTDIR/install.log  2>&1
    export PATH=$CURRENTDIR/deps/qcdnum/bin/:$PATH

    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    make -j 9  >> $CURRENTDIR/install.log  2>&1
    if [[ $? != 0 ]]
    then
        echo "Error, check install.log for details"
        exit
    fi
    cd - >& /dev/null

  # numdiff
    numdiff -v >& /dev/null
    if [[ `echo $?` != "0" ]]; then
      numdiffver='5.9.0'
      echo "Installing numdiff $numdiffver..."
      # 17 May 2020: this link to download numdiff seems to be dead
      #numdiff_url=http://gnu.mirrors.pair.com/savannah/savannah/numdiff/numdiff-${numdiffver}.tar.gz
      numdiff_url=http://ftp.igh.cnrs.fr/pub/nongnu/numdiff/numdiff-${numdiffver}.tar.gz
      if [[ $http == "curl" ]]
      then
          curl ${numdiff_url} > qcdnum${qcdnumstripver}.tar.gz 2>> $CURRENTDIR/install.log
      else
          wget ${numdiff_url} >> $CURRENTDIR/install.log 2>&1
      fi
      tar xfz numdiff-${numdiffver}.tar.gz  >> $CURRENTDIR/install.log  2>&1
      cd numdiff-${numdiffver}
      ./configure  --prefix=$CURRENTDIR/deps/numdiff  >> $CURRENTDIR/install.log  2>&1
      echo export PATH=$CURRENTDIR/deps/numdiff/bin/:\$PATH >> $CURRENTDIR/setup.sh
      if [[ $? != 0 ]]
      then
          echo "Error, check install.log for details"
          exit
      fi
      make -j 9  >> $CURRENTDIR/install.log  2>&1
      if [[ $? != 0 ]]
      then
          echo "Error, check install.log for details"
          exit
      fi
      cd - >& /dev/null
    fi
fi
cd $CURRENTDIR

 #xfitter
if [[ $mode != "deps" ]]
then
    echo "Installing xFitter $version..."



#else
#    make -C xfitter-${version} clean >> $CURRENTDIR/install.log  2>&1
fi


#make a setup run enviroment script
echo $version > version
echo "export CURRENTDIR=`pwd`" >> setup.sh
echo "export version=\`cat version\`" >> setup.sh
echo "export PATH=$CURRENTDIR/xfitter-$version/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/hoppet/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/applgrid/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/lhapdf/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/apfel/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/mela/bin:\$PATH" >> setup.sh
echo "export PATH=$CURRENTDIR/deps/apfelxx/bin:\$PATH" >> setup.sh

echo `lhapdf-config --prefix`/lib*/python*/site-packages > /tmp/xf_py1234_a
echo "export PYTHONPATH=$PYTHONPATH:`cat /tmp/xf_py1234_a` " >> setup.sh
rm /tmp/xf_py1234_a

echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/hoppet/lib/:\$LD_LIBRARY_PATH   >> setup.sh
echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/lhapdf/lib/:\$LD_LIBRARY_PATH   >> setup.sh
# echo export HATHOR_DIR=$CURRENTDIR/deps/Hathor-${hathorver}       >> setup.sh
echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/applgrid/lib/:\$LD_LIBRARY_PATH >> setup.sh
echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/apfel/lib/:\$LD_LIBRARY_PATH >> setup.sh
echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/mela/lib/:\$LD_LIBRARY_PATH >> setup.sh
echo export LD_LIBRARY_PATH=\$CURRENTDIR/deps/qcdnum/lib/:\$LD_LIBRARY_PATH >> setup.sh
echo "export PATH=$CURRENTDIR/deps/qcdnum/bin:\$PATH" >> setup.sh


if [[ $MODE == "cern" ]]
then
#    echo . /afs/cern.ch/sw/lcg/contrib/gcc/${gccv}/${arch}-${os}/setup.sh            >> setup.sh
    echo . /cvmfs/sft.cern.ch/lcg/contrib/gcc/${gccv}/${arch}-${os}/setup.sh            >> setup.sh
#    . /afs/cern.ch/sw/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root/bin/thisroot.sh
#    echo cd /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root >> setup.sh

    echo "cd /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root/" >> setup.sh
    echo ". /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/${rootversion}/${arch}-${os}-${compiler}-opt/root/bin/thisroot.sh ">> setup.sh
    echo "cd -" >>setup.sh
fi

#make a compilation script
echo source ./setup.sh > compile
echo export PATH=\$CURRENTDIR/deps/hoppet/bin/:\$PATH       >> compile
echo export PATH=\$CURRENTDIR/deps/lhapdf/bin/:\$PATH       >> compile
# echo export HATHOR_DIR=$CURRENTDIR/deps/Hathor-${hathorver}       >> compile
echo export PATH=\$CURRENTDIR/deps/applgrid/bin/:\$PATH     >> compile
echo export PATH=\$CURRENTDIR/deps/apfel/bin/:\$PATH     >> compile
echo export PATH=\$CURRENTDIR/deps/mela/bin/:\$PATH     >> compile


if [[ $mode != "deps" ]]
then
  echo cd xfitter-\$version                                        >> compile
fi

# configure
# if [[ $mode != "deps" ]]
# then
#   if [[ $version == "2.0.0" ]] || [[ $version == "2.0.1" ]]; then
#     echo autoreconf --install                                 >> compile
#     echo ./configure --enable-applgrid --enable-lhapdf --enable-apfel --enable-apfelxx --enable-mela  --enable-apfelgrid --enable-process    >> compile
#     echo make -j 9                                    >> compile
#   elif [[ $version == "master_before_PionCeres_merge" ]] || [[ $version == "test_ceres_v0.01" ]]; then
#     # SZ 10.07.2019 apfelgrid produces linking error in PionCeres
#     echo ./configure --enable-applgrid --enable-lhapdf --enable-hathor --enable-apfel --enable-apfelxx --enable-mela --enable-process    >> compile
#     echo make -j 9                                    >> compile
#     echo autoreconf --install                                 >> compile
#   else
#     # cmake compilation
#     echo ./make.sh install >> compile
#   fi
#   echo "Compiling xFitter $version..."
# fi

# ./compile >> $CURRENTDIR/install.log  2>&1
# if [[ $? != 0 ]]
# then
#     echo "Error, check install.log for details"
#     exit
# fi

source ./setup.sh

if [[ $mode == "deps" ]]
then
  echo "Installing xfitter dependencies is complete"
  echo "Check install.log file for details"
else
  # run test
  cd xfitter-${version}
  bin/xfitter >> $CURRENTDIR/install.log  2>&1

  if [[ $? != 0 ]]
  then
      echo "Error in testing xfitter executable, check install.log for details"
      exit
  fi
  cd - >& /dev/null
  echo "xFitter installation successful!"
  echo "Check install.log file for details"
  echo
  # setup a run dir
  if [[ ! -e run ]]
  then
      mkdir -p run
      cp  xfitter-${version}/steering.txt \
          xfitter-${version}/parameters.yaml \
          xfitter-${version}/constants.yaml \
          run
      rsync -a --exclude=".*" xfitter-${version}/datafiles run/
      rsync -a --exclude=".*" xfitter-${version}/theoryfiles run/
  else
      echo "\"run\" directory already exists, I won't touch it"
      echo
  fi

  
fi
