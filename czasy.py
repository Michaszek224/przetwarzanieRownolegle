import matplotlib.pyplot as plt
import numpy as np

def porownaj_i_wykresl_czasy(num_points, czasy_gpu, czasy_cpu, czasy_openmp):
    """
    Porównuje czasy wykonania dla GPU, CPU i OpenMP i generuje wykres.

    Args:
        num_points (list): Lista liczb punktów, dla których mierzone były czasy.
        czasy_gpu (list): Lista czasów wykonania dla GPU.
        czasy_cpu (list): Lista czasów wykonania dla CPU.
        czasy_openmp (list): Lista czasów wykonania dla OpenMP.
    """

    num_points_cpu = num_points[:-1]
    czasy_cpu_clean = czasy_cpu[:-1]

    plt.figure(figsize=(12, 7))
    plt.plot(num_points, czasy_gpu, marker='o', linestyle='-', label='GPU')
    plt.plot(num_points_cpu, czasy_cpu_clean, marker='o', linestyle='-', label='CPU')
    plt.plot(num_points, czasy_openmp, marker='o', linestyle='-', label='OpenMP')

    plt.xscale('log')

    plt.title('Porównanie czasów wykonania (GPU vs CPU vs OpenMP)')
    plt.xlabel('Liczba punktów')
    plt.ylabel('Czas wykonania (sekundy)')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.savefig('porownanie_czasow_gpu_cpu_openmp.png')

num_points = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
czasy_gpu = [0.0738, 0.0704, 0.0625, 0.1039, 0.2047, 1.2642, 13.1729]
czasy_cpu = [0.000109, 0.000579, 0.007703, 0.401321, 1.052377, 90.350555, '-']
czasy_openmp = [0.0143, 0.0146, 0.0130, 0.0433, 0.2888, 3.4810, 31.2816]

porownaj_i_wykresl_czasy(num_points, czasy_gpu, czasy_cpu, czasy_openmp)