import numpy as np
from sklearn.datasets import make_blobs
import os
import time

# --- Parametry generowania danych ---
num_points = 10000000 # Liczba punktów danych (jak w Twoim przykładzie)
dimensions = 2        # Liczba wymiarów (dla wizualizacji najlepiej 2)
num_clusters_in_data = 5 # Liczba naturalnych klastrów w generowanych danych
std_dev_of_clusters = 1.0 # Odchylenie standardowe klastrów (mniejsze = bardziej zwarte klastry)
filename = "data.txt"

print(f"Rozpoczynam generowanie {num_points} punktów o {dimensions} wymiarach przy użyciu make_blobs...")
start_time = time.time()

# Generowanie punktów danych za pomocą make_blobs
data_points, _ = make_blobs(n_samples=num_points,
                            n_features=dimensions,
                            centers=num_clusters_in_data,
                            cluster_std=std_dev_of_clusters,
                            random_state=42)

end_time = time.time()
print(f"Generowanie danych zakończono w {end_time - start_time:.2f} sekundy.")

print(f"Zapisywanie wygenerowanych punktów do pliku: {filename}...")
start_time = time.time()

# --- MODYFIKACJA TUTAJ ---
# Otwórz plik w trybie zapisu
with open(filename, 'w') as f:
    # Najpierw zapisz liczbę punktów i wymiary
    f.write(f"{num_points} {dimensions}\n")
    # Następnie zapisz dane punktów
    # Użyj np.savetxt z mode='a' aby dopisać do pliku, ale lepiej to zrobić ręcznie
    # lub po prostu użyć genfromtxt w C, który nie wymaga headera
    # Tutaj zapisujemy w pętli dla prostoty, aby mieć pewność, że wszystko jest w jednym pliku
    for row in data_points:
        f.write(" ".join(map(lambda x: f"{x:.6f}", row)) + "\n")

end_time = time.time()
print(f"Zapis danych zakończono w {end_time - start_time:.2f} sekundy.")
print(f"Wygenerowano {num_points} punktów o {dimensions} wymiarach do pliku {filename}")