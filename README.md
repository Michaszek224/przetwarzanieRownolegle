# To generate points #
python3 main.py

# To compile #
gcc -O3 -o kmeans_gpu gpu.c -lOpenCL -lm
<br>
gcc -o kmeans_sekwencyjny sekwencyjny.c -lm
<br>
gcc openmp.c -o kmeans_openmp -lm -fopenmp


# To run #
./kmeans_gpu data.txt 5 100
<br>
./kmeans_sekwencyjny data.txt 5 100
<br>
./kmeans_openmp data.txt 5 100


# To plot #
python3 wizu.py
