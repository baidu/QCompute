ubuntu 18.04
设置快速的apt源
更新系统到最新

【如cmake版本<3.16 安装cmake】
推荐官网上的源码编译方式


【安装mkl】
sudo bash
# <type your user password when prompted.  this will put you in a root shell>
# cd to /tmp where this shell has write permission
cd /tmp
# now get the key:
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now install that key
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now remove the public key file exit the root shell
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
exit
sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo sh -c 'echo deb https://apt.repos.intel.com/ipp all main > /etc/apt/sources.list.d/intel-ipp.list'
sudo sh -c 'echo deb https://apt.repos.intel.com/tbb all main > /etc/apt/sources.list.d/intel-tbb.list'
sudo sh -c 'echo deb https://apt.repos.intel.com/daal all main > /etc/apt/sources.list.d/intel-daal.list'
sudo sh -c 'echo deb https://apt.repos.intel.com/mpi all main > /etc/apt/sources.list.d/intel-mpi.list'
sudo sh -c 'echo deb https://apt.repos.intel.com/intelpython binary/ > /etc/apt/sources.list.d/intelpython.list'
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install intel-mkl-2020.0-088


【设置mkl路径】
在.bashrc中追加
source /opt/intel/mkl/bin/mklvars.sh intel64
设置路径脚本仅对当前bash生效，并非永久设置，故需写入.bashrc


【安装boost及组件库】
sudo apt-get update
sudo apt-get install libboost-dev
sudo apt-get install libboost-program-options-dev
sudo apt-get install libboost-date-time-dev
sudo apt-get install libboost-thread-dev
sudo apt-get install libboost-filesystem-dev
sudo apt-get install libboost-chrono-dev

【安装python headers】
sudo apt-get install libpython3.9-dev

【编译】
cd SymmetricQSPInternalCpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
执行冒烟测试脚本，一定要测试，一些库的故障能通过build，但是执行出错
