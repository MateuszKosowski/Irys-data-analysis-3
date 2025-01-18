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

# Funkcja do analizy klasyfikacji k-NN
def analyze_knn(feature_columns, description):

    # Oddzielenie danych od etykiet
    x_train = data_train[feature_columns].values
    y_train = data_train['Gatunek'].values

    x_test = data_test[feature_columns].values
    y_test_f = data_test['Gatunek'].values

    # Normalizacja danych
    x_train_norm = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_test_norm = (x_test - x_test.min()) / (x_test.max() - x_test.min())

    # Lista do przechowywania dokładności modelu dla różnych wartości k
    accuracies = []

    # Zakres k
    k_range = range(1, 16)

    # Sprawdzenie dokładności modelu dla różnych wartości k
    for k in k_range:

        # Tworzenie i trenowanie modelu k-NN z wagami opartymi na odległości aby rozwiązać problem remisów
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(x_train_norm, y_train)

        # Sprawdzenie dokładności modelu
        y_pred = knn.predict(x_test_norm)
        accuracy = accuracy_score(y_test_f, y_pred) # accuracy zwraca od 0 do 1, gdzie 1 oznacza 100% dokładności
        accuracies.append(accuracy)

    # Wykres dokładności
    plt.figure(figsize=(9, 6), tight_layout=True)
    plt.bar(k_range, accuracies)
    plt.title(description, fontsize=16, pad=15)
    plt.xlabel('Liczba sąsiadów (k)', fontsize=14, labelpad=15)
    plt.ylabel('Dokładność', fontsize=14, labelpad=15)
    plt.xticks(k_range, fontsize=12)
    plt.grid(axis='y', which='both')
    plt.ylim(0.60, 1.05)
    plt.yticks(np.arange(0.60, 1.05, 0.05), fontsize=12)
    plt.show()

    # Najlepsza wartość k
    print(f'--- Wyniki dla cech {feature_columns} ---')
    best_k = k_range[np.argmax(accuracies)]
    print(f'Najlepsze k: {best_k}, dokładność: {max(accuracies):.2f}')

    # Macierz pomyłek dla najlepszego k
    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn.fit(x_train_norm, y_train)
    y_pred = knn.predict(x_test_norm)
    conf_matrix = confusion_matrix(y_test_f, y_pred) # Macierz pomyłek
    print('Macierz pomyłek dla najlepszego k:')
    print(conf_matrix)

# Wywołanie funkcji
analyze_knn(['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka'], 'Dokładność klasyfikacji dla różnych k - wszystkie cechy')
analyze_knn(['Dlugosc kielicha', 'Szerokosc kielicha'], 'Dokładność klasyfikacji dla różnych k - długość i szerokość kielicha')
analyze_knn(['Dlugosc platka', 'Szerokosc platka'], 'Dokładność klasyfikacji dla różnych k - długość i szerokość płatka')
analyze_knn(['Dlugosc kielicha', 'Dlugosc platka'], 'Dokładność klasyfikacji dla różnych k - długość kielicha i płatka')
analyze_knn(['Szerokosc kielicha', 'Szerokosc platka'], 'Dokładność klasyfikacji dla różnych k - szerokość kielicha i płatka')
analyze_knn(['Dlugosc kielicha', 'Szerokosc platka'], 'Dokładność klasyfikacji dla różnych k - długość kielicha i szerokość płatka')
analyze_knn(['Szerokosc kielicha', 'Dlugosc platka'], 'Dokładność klasyfikacji dla różnych k - szerokość kielicha i długość płatka')

