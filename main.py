import numpy as np
from sklearn.datasets import make_blobs
import os
import time

num_points = 10000000
dimensions = 2
num_clusters_in_data = 5
std_dev_of_clusters = 1.0
filename = "data.txt"

print(f"Rozpoczynam generowanie {num_points} punktów o {dimensions} wymiarach przy użyciu make_blobs...")
start_time = time.time()

data_points, _ = make_blobs(n_samples=num_points,
                            n_features=dimensions,
                            centers=num_clusters_in_data,
                            cluster_std=std_dev_of_clusters,
                            random_state=42)

end_time = time.time()
print(f"Generowanie danych zakończono w {end_time - start_time:.2f} sekundy.")

print(f"Zapisywanie wygenerowanych punktów do pliku: {filename}...")
start_time = time.time()

with open(filename, 'w') as f:
    f.write(f"{num_points} {dimensions}\n")
    for row in data_points:
        f.write(" ".join(map(lambda x: f"{x:.6f}", row)) + "\n")

end_time = time.time()
print(f"Zapis danych zakończono w {end_time - start_time:.2f} sekundy.")
print(f"Wygenerowano {num_points} punktów o {dimensions} wymiarach do pliku {filename}")
