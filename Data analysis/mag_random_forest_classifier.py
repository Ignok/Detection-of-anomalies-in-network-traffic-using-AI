import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Wczytanie danych
data_train = pd.read_csv('./Data/UNSW_NB15_training-set.csv')
data_test = pd.read_csv('./Data/UNSW_NB15_testing-set.csv')



### Label as results ###

"""
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

# 1. Model Random Forest z domyślnymi parametrami
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled)

# Obliczanie metryk
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Zapis wyników do pliku
save_results_to_file("random_forest_label_results.txt", "Random Forest with default parameters:")
save_results_to_file("random_forest_label_results.txt", f"Accuracy: {accuracy:.2f}")
save_results_to_file("random_forest_label_results.txt", f"Precision: {precision:.2f}")
save_results_to_file("random_forest_label_results.txt", f"Recall: {recall:.2f}")
save_results_to_file("random_forest_label_results.txt", f"F1 Score: {f1:.2f}")
save_results_to_file("random_forest_label_results.txt", "\n" + classification_report(y_test, y_pred))

# 2. Optymalizacja hiperparametrów za pomocą GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Najlepsze parametry i model
best_params = grid_search.best_params_
best_rf_classifier = grid_search.best_estimator_

# Predykcje na zbiorze testowym
y_pred_best = best_rf_classifier.predict(X_test_scaled)

# Obliczanie metryk dla najlepszego modelu
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

# Zapis wyników do pliku
save_results_to_file("random_forest_label_results.txt", "\nRandom Forest with optimized parameters:")
save_results_to_file("random_forest_label_results.txt", f"Best Parameters: {best_params}")
save_results_to_file("random_forest_label_results.txt", f"Accuracy: {accuracy_best:.2f}")
save_results_to_file("random_forest_label_results.txt", f"Precision: {precision_best:.2f}")
save_results_to_file("random_forest_label_results.txt", f"Recall: {recall_best:.2f}")
save_results_to_file("random_forest_label_results.txt", f"F1 Score: {f1_best:.2f}")
save_results_to_file("random_forest_label_results.txt", "\n" + classification_report(y_test, y_pred_best))

print("Random Forest Classifier completed. Results are saved in random_forest_label_results.txt.")

"""

### Attack_cat as results ###

# Kopia danych treningowych
x_train = data_train.copy()
y_train = data_train['attack_cat']  # Użycie kolumny 'attack_cat' jako wynikowej

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
y_test = data_test['attack_cat']  # Użycie kolumny 'attack_cat' jako wynikowej

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

# 1. Model Random Forest z domyślnymi parametrami
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train_encoded)

y_pred = rf_classifier.predict(X_test_scaled)

# Obliczanie metryk
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

# Zapis wyników do pliku
save_results_to_file("random_forest_attack_cat_results.txt", "Random Forest with default parameters:")
save_results_to_file("random_forest_attack_cat_results.txt", f"Accuracy: {accuracy:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"Precision: {precision:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"Recall: {recall:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"F1 Score: {f1:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", "\n" + classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# 2. Optymalizacja hiperparametrów za pomocą GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_encoded)

# Najlepsze parametry i model
best_params = grid_search.best_params_
best_rf_classifier = grid_search.best_estimator_

# Predykcje na zbiorze testowym
y_pred_best = best_rf_classifier.predict(X_test_scaled)

# Obliczanie metryk dla najlepszego modelu
accuracy_best = accuracy_score(y_test_encoded, y_pred_best)
precision_best = precision_score(y_test_encoded, y_pred_best, average='weighted')
recall_best = recall_score(y_test_encoded, y_pred_best, average='weighted')
f1_best = f1_score(y_test_encoded, y_pred_best, average='weighted')

# Zapis wyników do pliku
save_results_to_file("random_forest_attack_cat_results.txt", "\nRandom Forest with optimized parameters:")
save_results_to_file("random_forest_attack_cat_results.txt", f"Best Parameters: {best_params}")
save_results_to_file("random_forest_attack_cat_results.txt", f"Accuracy: {accuracy_best:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"Precision: {precision_best:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"Recall: {recall_best:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", f"F1 Score: {f1_best:.2f}")
save_results_to_file("random_forest_attack_cat_results.txt", "\n" + classification_report(y_test_encoded, y_pred_best, target_names=le.classes_))

print("Random Forest Classifier for 'attack_cat' completed. Results are saved in random_forest_attack_cat_results.txt.")