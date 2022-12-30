确保cmake 版本>3.6
下源码。
./bootstrap
make
make install


yum装boost
yum install boost boost-devel boost-doc



yum装mkl
https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-yum-repo.html
sudo yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo
yum install intel-mkl-64bit-2020.0-088

替换修改cmakelists.txt(现在那个给云帆修改的不配套)


有py虚拟环境的，进虚拟环境
进SimulatorCppSrc目录下编译
mkdir build
cd build 
cmake -DCMAKE_BUILD_TYPE=Release ..  
make 
成功后会在SimulatorCppSrc上一层多个新的so文件
如果报错缺python.h, yum装对应的python3-devel


把孟泽霖的测试文件放在example下，开头增加
import sys
sys.path.append("../..")
以识别库的主目录，执行
