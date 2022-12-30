macos 12.6 intel虚拟机
macos xx.x M1 真机


【安装brew】
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

【cmake】
brew install cmake

【重装brew版 3.9】
// see https://stackoverflow.com/questions/32578106/how-to-install-python-devel-in-mac-os
brew reinstall python@3.9
输入which python检查应该返回类似：/usr/bin/python3
不要用原装py 3.9，会找不到解释器
不能用miniconda/anaconda，否则后面会找不到头文件，如果有虚拟环境请退出


//【安装libomp(没调对)】
//brew install libomp
//export CC=/usr/bin/clang
//export CXX=/usr/bin/clang++
//export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
//export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
//export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
//export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"

【把源码中有关MKL和OPENMP加速的删除】

【把cmakelist中有关UNIX下的static-libgcc的一整块删除，macos不适用】

//【验证一下Python headers】
// find / -name Python.h 2>/dev/null
// /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers/Python.h
// /System/Volumes/Data/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers/Python.h
// 修改cmake插入搜索路径
// pip install pybind11?


【编译】
cd SymmetricQSPInternalCpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
执行冒烟测试脚本，一定要测试，一些库的故障能通过build，但是执行出错
