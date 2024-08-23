from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# Wczytanie danych
data_train = pd.read_csv('./Data/UNSW_NB15_training-set.csv')
data_test = pd.read_csv('./Data/UNSW_NB15_testing-set.csv')

### Label as results ###

# Kopia danych treningowych
x_train = data_train.copy()
y_train = data_train['label']  # Użycie kolumny 'label' jako wynikowej

# Przetwarzanie danych nominalnych
nominal_cols = ['proto', 'service', 'state']

# OneHotEncoder
onehot = OneHotEncoder(dtype=int, sparse_output=False, handle_unknown='ignore')

# Dopasowanie i przekształcenie danych nominalnych
encoded_nominals = onehot.fit_transform(x_train[nominal_cols])

# Tworzenie DataFrame z zakodowanymi kolumnami
encoded_nominals_df = pd.DataFrame(encoded_nominals, columns=onehot.get_feature_names_out(nominal_cols))

# Usunięcie oryginalnych kolumn nominalnych z DataFrame
x_train = x_train.drop(columns=nominal_cols)
x_train = x_train.drop(columns=["id"])

# Połączenie zakodowanych kolumn z oryginalnym DataFrame
x_train = pd.concat([x_train, encoded_nominals_df], axis=1)

# PowerTransformer
cols_to_scale = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
                 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                 'ct_src_ltm', 'ct_srv_dst']

# Box-Cox Transformation
pt = PowerTransformer(method='box-cox')
x_train[cols_to_scale] = pt.fit_transform(x_train[cols_to_scale] + 1)
scaled_x_train = pd.concat([x_train[cols_to_scale], x_train.drop(columns=cols_to_scale)], axis=1)

# Normalizacja datasetu testowego
x_test = data_test.copy()
y_test = data_test['label']  # Użycie kolumny 'label' jako wynikowej

# Kodowanie danych nominalnych w zbiorze testowym
encoded_nominals_test = onehot.transform(x_test[nominal_cols])
encoded_nominals_test_df = pd.DataFrame(encoded_nominals_test, columns=onehot.get_feature_names_out(nominal_cols))

# Usunięcie oryginalnych kolumn nominalnych i kolumny 'id' z DataFrame
x_test = x_test.drop(columns=nominal_cols + ["id"])

# Połączenie zakodowanych kolumn z oryginalnym DataFrame
x_test = pd.concat([x_test, encoded_nominals_test_df], axis=1)

# Skalowanie danych testowych przy użyciu tych samych parametrów co dla zbioru treningowego
x_test[cols_to_scale] = pt.transform(x_test[cols_to_scale] + 1)

# Dodanie brakujących kolumn do zbiorów danych
missing_cols_train = x_test.columns.difference(x_train.columns)
x_train = x_train.reindex(columns=x_train.columns.union(missing_cols_train), fill_value=0)

missing_cols_test = x_train.columns.difference(x_test.columns)
x_test = x_test.reindex(columns=x_test.columns.union(missing_cols_test), fill_value=0)

# Przygotowanie danych wejściowych
X_train = x_train.drop(columns=['label', 'attack_cat'])
X_test = x_test.drop(columns=['label', 'attack_cat'])

# Normalizacja danych wejściowych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding dla y_train i y_test
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Funkcja do zapisu wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

# 1. Model KMeans z domyślnymi parametrami
kmeans_default = KMeans(random_state=42)
kmeans_default.fit(X_train_scaled)

# Predykcje i ocena modelu
y_train_pred = kmeans_default.predict(X_train_scaled)
silhouette_avg = silhouette_score(X_train_scaled, y_train_pred)
adjusted_rand = adjusted_rand_score(y_train_encoded, y_train_pred)

# Zapis wyników do pliku
save_results_to_file("kmeans_label_results.txt", "KMeans with default parameters:")
save_results_to_file("kmeans_label_results.txt", f"Silhouette Score: {silhouette_avg}")
save_results_to_file("kmeans_label_results.txt", f"Adjusted Rand Index: {adjusted_rand}")

# 2. Szukanie najlepszej liczby klastrów
best_silhouette = -1
best_k = 2
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    y_train_pred = kmeans.predict(X_train_scaled)
    silhouette_avg = silhouette_score(X_train_scaled, y_train_pred)
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_k = k

# Budowanie modelu z najlepszą liczbą klastrów
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
kmeans_best.fit(X_train_scaled)
y_train_pred_best = kmeans_best.predict(X_train_scaled)
best_adjusted_rand = adjusted_rand_score(y_train_encoded, y_train_pred_best)

# Zapis wyników do pliku
save_results_to_file("kmeans_label_results.txt", "\nKMeans with optimized parameters:")
save_results_to_file("kmeans_label_results.txt", f"Best Number of Clusters: {best_k}")
save_results_to_file("kmeans_label_results.txt", f"Best Silhouette Score: {best_silhouette}")
save_results_to_file("kmeans_label_results.txt", f"Adjusted Rand Index with best k: {best_adjusted_rand}")

print("KMeans Clustering with 'label' completed. Results are saved in kmeans_label_results.txt.")



### Attack_cat as results ###
"""
# Kopia danych treningowych
x_train = data_train.copy()
y_train = data_train['attack_cat']

# Przetwarzanie danych nominalnych
nominal_cols = ['proto', 'service', 'state']

# OneHotEncoder
onehot = OneHotEncoder(dtype=int, sparse_output=False, handle_unknown='ignore')

# Dopasowanie i przekształcenie danych nominalnych
encoded_nominals = onehot.fit_transform(x_train[nominal_cols])

# Tworzenie DataFrame z zakodowanymi kolumnami
encoded_nominals_df = pd.DataFrame(encoded_nominals, columns=onehot.get_feature_names_out(nominal_cols))

# Usunięcie oryginalnych kolumn nominalnych z DataFrame
x_train = x_train.drop(columns=nominal_cols)
x_train = x_train.drop(columns=["id"])

# Połączenie zakodowanych kolumn z oryginalnym DataFrame
x_train = pd.concat([x_train, encoded_nominals_df], axis=1)

# PowerTransformer
cols_to_scale = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
                 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                 'ct_src_ltm', 'ct_srv_dst']

# Box-Cox Transformation
pt = PowerTransformer(method='box-cox')
x_train[cols_to_scale] = pt.fit_transform(x_train[cols_to_scale] + 1)
scaled_x_train = pd.concat([x_train[cols_to_scale], x_train.drop(columns=cols_to_scale)], axis=1)

# Normalizacja datasetu testowego
x_test = data_test.copy()
y_test = data_test['attack_cat']

# Kodowanie danych nominalnych w zbiorze testowym
encoded_nominals_test = onehot.transform(x_test[nominal_cols])
encoded_nominals_test_df = pd.DataFrame(encoded_nominals_test, columns=onehot.get_feature_names_out(nominal_cols))

# Usunięcie oryginalnych kolumn nominalnych i kolumny 'id' z DataFrame
x_test = x_test.drop(columns=nominal_cols + ["id"])

# Połączenie zakodowanych kolumn z oryginalnym DataFrame
x_test = pd.concat([x_test, encoded_nominals_test_df], axis=1)

# Skalowanie danych testowych przy użyciu tych samych parametrów co dla zbioru treningowego
x_test[cols_to_scale] = pt.transform(x_test[cols_to_scale] + 1)

# Dodanie brakujących kolumn do zbiorów danych
missing_cols_train = x_test.columns.difference(x_train.columns)
x_train = x_train.reindex(columns=x_train.columns.union(missing_cols_train), fill_value=0)

missing_cols_test = x_train.columns.difference(x_test.columns)
x_test = x_test.reindex(columns=x_test.columns.union(missing_cols_test), fill_value=0)

# Przygotowanie danych wejściowych
X_train = x_train.drop(columns=['label', 'attack_cat'])
X_test = x_test.drop(columns=['label', 'attack_cat'])

# Normalizacja danych wejściowych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding dla y_train i y_test
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Funkcja do zapisu wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

# 1. Model KMeans z domyślnymi parametrami
kmeans_default = KMeans(random_state=42)
kmeans_default.fit(X_train_scaled)

# Predykcje i ocena modelu
y_train_pred = kmeans_default.predict(X_train_scaled)
silhouette_avg = silhouette_score(X_train_scaled, y_train_pred)
adjusted_rand = adjusted_rand_score(y_train_encoded, y_train_pred)

# Zapis wyników do pliku
save_results_to_file("kmeans_results.txt", "KMeans with default parameters:")
save_results_to_file("kmeans_results.txt", f"Silhouette Score: {silhouette_avg}")
save_results_to_file("kmeans_results.txt", f"Adjusted Rand Index: {adjusted_rand}")

# 2. Szukanie najlepszej liczby klastrów
best_silhouette = -1
best_k = 2
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    y_train_pred = kmeans.predict(X_train_scaled)
    silhouette_avg = silhouette_score(X_train_scaled, y_train_pred)
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_k = k

# Budowanie modelu z najlepszą liczbą klastrów
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
kmeans_best.fit(X_train_scaled)
y_train_pred_best = kmeans_best.predict(X_train_scaled)
best_adjusted_rand = adjusted_rand_score(y_train_encoded, y_train_pred_best)

# Zapis wyników do pliku
save_results_to_file("kmeans_results.txt", "\nKMeans with optimized parameters:")
save_results_to_file("kmeans_results.txt", f"Best Number of Clusters: {best_k}")
save_results_to_file("kmeans_results.txt", f"Best Silhouette Score: {best_silhouette}")
save_results_to_file("kmeans_results.txt", f"Adjusted Rand Index with best k: {best_adjusted_rand}")

print("KMeans Clustering completed. Results are saved in kmeans_results.txt.")


"""

