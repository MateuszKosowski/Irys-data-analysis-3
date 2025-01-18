import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from numpy.matrixlib.defmatrix import matrix
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

