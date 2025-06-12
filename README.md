# To generate points #
python3 main.py

# To compile #
gcc -O3 -o kmeans_gpu gpu.c -lOpenCL -lm
<br>
gcc -o kmeans_cpu cpu.c -lm

# To run #
./kmeans_gpu data.txt 5 100
<br>
./kmeans_cpu data.txt 5 100

# To plot #
python3 wizu.py
