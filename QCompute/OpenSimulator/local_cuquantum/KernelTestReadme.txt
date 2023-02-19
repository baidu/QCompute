This is the instructions for using the standalone version of the simulator based on cucquantum

1. install cuda driver
Install driver by nvidia's run file or ubuntu's apt. We use ubuntu 20.4 and nvidia driver 470.141.03.
Install cuda toolkit 11.4 by nvidia's run file

2. install libs
python 3.8 is suggested
sudo apt install libopenmpi-dev
pip install mpi4py==3.1.3
pip install cupy-cuda114
pip install cuquantum-python-cu11==22.7.1

3. run the script
run KernelTest.py

4. output
---------- init ----------
---------- U [3.5764134230726126, 5.040780044795463, 0.3965118560676544] [3] ----------
---------- CX [24, 14] ----------
---------- CX [20, 12] ----------
... 300 random CX/U gates ...
---------- U [4.644574037907133, 3.647684444050852, 2.8344305808667176] [4] ----------
---------- measure ----------
{'00001110111000111010011110': 1, '01010011011011101011001010': 1, ..., '00101010000010110000100010': 1}
