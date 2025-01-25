import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

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
def analyze_knn(feature_columns, description, figure_number):

    # Oddzielenie danych od etykiet
    x_train = data_train[feature_columns].values
    y_train = data_train['Gatunek'].values

    x_test = data_test[feature_columns].values
    y_test = data_test['Gatunek'].values

    # Normalizacja danych
    scaler = MinMaxScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm= scaler.transform(x_test) # Dane testowe są normalizowane zgodnie z tym samym odniesieniem, co dane treningowe, co zapewnia spójność.

    # Lista do przechowywania dokładności modelu dla różnych wartości k
    accuracies = []

    # Zakres k
    k_range = range(1, 16)

    # Sprawdzenie dokładności modelu dla różnych wartości k
    for k in k_range:

        # Tworzenie i trenowanie modelu k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_norm, y_train)

        # Sprawdzenie dokładności modelu
        y_pred = knn.predict(x_test_norm)
        accuracy = accuracy_score(y_test, y_pred) # accuracy zwraca od 0 do 1, gdzie 1 oznacza 100% dokładności
        accuracies.append(accuracy * 100)


    # Wykres dokładności
    plt.figure(figure_number, figsize=(10, 6), tight_layout=True)
    plt.bar(k_range, accuracies)
    plt.title(description, fontsize=16, pad=15)
    plt.xlabel('Liczba sasiadow (k)', fontsize=16, labelpad=15)
    plt.ylabel('Dokladnosc (%)', fontsize=16, labelpad=15)
    plt.xticks(k_range, fontsize=16)
    plt.grid(axis='y', which='both')
    plt.ylim(60, 105)
    plt.yticks(np.arange(60, 105, 5), fontsize=16)

    # Najlepsza wartość k
    print(f'--- Wyniki dla cech {feature_columns} ---')
    best_k = k_range[np.argmax(accuracies)]  # Zwraca indeks maksymalnej wartości z listy accuracies, jeśli są remisy to zwraca pierwszy
    print(f'Najlepsze k: {best_k}, dokladnosc: {max(accuracies):.2f}%')

    # Macierz pomyłek dla najlepszego k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train_norm, y_train)
    y_pred = knn.predict(x_test_norm)
    conf_matrix = confusion_matrix(y_test, y_pred) # Macierz pomyłek
    print('Macierz pomylek dla najlepszego k:')
    print(conf_matrix)

# Wywolanie funkcji
analyze_knn(['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka'], 'Dokladnosc klasyfikacji dla roznych k - wszystkie cechy', 1)
analyze_knn(['Dlugosc kielicha', 'Szerokosc kielicha'], 'Dokladnosc klasyfikacji dla roznych k - dlugosc i szerokosc kielicha', 2)
analyze_knn(['Dlugosc platka', 'Szerokosc platka'], 'Dokladnosc klasyfikacji dla roznych k - dlugosc i szerokosc platka', 3)
analyze_knn(['Dlugosc kielicha', 'Dlugosc platka'], 'Dokladnosc klasyfikacji dla roznych k - dlugosc kielicha i platka', 4)
analyze_knn(['Szerokosc kielicha', 'Szerokosc platka'], 'Dokladnosc klasyfikacji dla roznych k - szerokosc kielicha i platka', 5)
analyze_knn(['Dlugosc kielicha', 'Szerokosc platka'], 'Dokladnosc klasyfikacji dla roznych k - dlugosc kielicha i szerokosc platka', 6)
analyze_knn(['Szerokosc kielicha', 'Dlugosc platka'], 'Dokladnosc klasyfikacji dla roznych k - szerokosc kielicha i dlugosc platka', 7)

plt.show()

