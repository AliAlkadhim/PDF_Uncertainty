export CURRENTDIR=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CURRENTDIR/deps-yaml/yaml/lib/:$CURRENTDIR/deps-yaml/yaml-cpp/lib
export CFLAGS="$CFLAGS -I$CURRENTDIR/deps-yaml/yaml/include"
export CPPFLAGS="$CPPGLAGS -I$CURRENTDIR/deps-yaml/yaml/include -I/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps-yaml/yaml-cpp/include "
export CXXFLAGS="$CXXFLAGS -I$CURRENTDIR/deps-yaml/yaml/include -I/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps-yaml/yaml-cpp/include "
export LDFLAGS="$LDFLAGS -L$CURRENTDIR/deps-yaml/yaml/lib -L/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps-yaml/yaml-cpp/lib -lyaml-cpp "
export yaml_cpp_DIR=/storage/5/home/aalkadhim/cms04/CMSSW_10_2_23/src/deps-yaml/yaml-cpp
