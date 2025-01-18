import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Wczytaj plik csv
file_test = './dane/data3_test.csv'
file_train = './dane/data3_train.csv'

# Dane do trenowania
data_train = pd.read_csv(file_train, header=None, sep=',')
data_train.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']

# Dane do testowania
data_test = pd.read_csv(file_test, header=None, sep=',')
data_test.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']

# Oddzielenie danych od etykiet
X_train = data_train.iloc[:, :-1].values # Wszystkie wiersze, wszystkie kolumny oprócz ostatniej
y_train = data_train.iloc[:, -1].values # Wszystkie wiersze, ostatnia kolumna

X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values

# Normalizacja danych
X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())

X_test_norm = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# Lista do przechowywania dokładności modelu dla różnych wartości k
k_range = range(1, 16)
accuracies = []

# Sprawdzenie dokładności modelu dla różnych wartości k
for k in k_range:

    # Tworzenie i trenowanie modelu k-NN z wagami opartymi na odległości
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_norm, y_train)

    # Sprawdzenie dokładności modelu
    y_pred = knn.predict(X_test_norm)
    accuracy = accuracy_score(y_test, y_pred) # accuracy zwraca od 0 do 1, gdzie 1 oznacza 100% dokładności
    accuracies.append(accuracy)

# Najlepsza wartość k
best_k = k_range[np.argmax(accuracies)] # Zwraca indeks wartości maksymalnej, jeśli jest kilka takich wartości, zwraca pierwszą
print(f'Najlepsze k: {best_k}, dokładność: {max(accuracies):.2f}')

# Macierz pomyłek dla najlepszego k
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn.fit(X_train_norm, y_train)
y_pred = knn.predict(X_test_norm)
conf_matrix = confusion_matrix(y_test, y_pred) # Macierz pomyłek
print('Macierz pomyłek dla najlepszego k:')
print(conf_matrix)

# Wykres dokładności
plt.figure(figsize=(9, 6), tight_layout=True)
plt.bar(k_range, accuracies)
plt.title('Dokładność klasyfikacji dla różnych k (wszystkie cechy)', fontsize=16, pad=15)
plt.xlabel('Liczba sąsiadów (k)', fontsize=14, labelpad=15)
plt.ylabel('Dokładność', fontsize=14, labelpad=15)
plt.xticks(k_range, fontsize=12)
plt.grid(axis='y', which='both')
plt.ylim(0.9, 1.02)
plt.yticks(np.arange(0.9, 1.02, 0.02), fontsize=12)
plt.show()

