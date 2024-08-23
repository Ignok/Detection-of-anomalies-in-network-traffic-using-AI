# -*- coding: utf-8 -*-

from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score
import optuna


# Wczytanie danych
data_train = pd.read_csv('./Data/UNSW_NB15_training-set.csv')
data_test = pd.read_csv('./Data/UNSW_NB15_testing-set.csv')

"""
"""
#### ATTACK CAT AS RESULTS ####

# Kopia danych treningowych
x_train = data_train.copy()
y_train = data_train.attack_cat

from sklearn.preprocessing import OneHotEncoder

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

from sklearn.preprocessing import StandardScaler
cols_to_scale = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
                 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                 'ct_src_ltm', 'ct_srv_dst']

# Separacja kolumn binarnych i klasyfikacyjnych
binary_cols = x_train.columns[x_train.columns.str.startswith(('proto_', 'service_', 'state_', 'is_ftp_login'))].tolist()
class_cols = ['label', 'attack_cat']

### StandardScaler

# Skalowanie wybranych kolumn
#scaler = StandardScaler()
#x_train[cols_to_scale] = scaler.fit_transform(x_train[cols_to_scale])

# Połączenie przeskalowanych danych z kolumnami binarnymi i klasyfikacyjnymi
#scaled_x_train = pd.concat([x_train[cols_to_scale], x_train[binary_cols], x_train[class_cols]], axis=1)


### Box-Cox Transformation

pt = PowerTransformer(method='box-cox')
x_train[cols_to_scale] = pt.fit_transform(x_train[cols_to_scale] + 1)
scaled_x_train = pd.concat([x_train[cols_to_scale],
                            x_train[binary_cols], x_train[class_cols]], axis=1)


### Przekształcenie logarytmiczne

# Skalowanie wybranych kolumn
#x_train[cols_to_scale] = np.log1p(x_train[cols_to_scale])
#scaled_x_train = pd.concat([x_train[cols_to_scale],
#                            x_train[binary_cols], x_train[class_cols]], axis=1)

# Normalizacja datasetu testowego
x_test = data_test.copy()
y_test = data_test.label

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

# # Zakodowanie kolumny "attack_cat" przy użyciu LabelEncoder
# le = LabelEncoder()
# x_train['attack_cat_encoded'] = le.fit_transform(x_train['attack_cat'])
# x_test['attack_cat_encoded'] = le.transform(x_test['attack_cat'])

# Przygotowanie danych wejściowych i wyjściowych
X_train = x_train.drop(columns=['label', 'attack_cat'])
y_train = x_train['attack_cat']
X_test = x_test.drop(columns=['label', 'attack_cat'])
y_test = x_test['attack_cat']

# Normalizacja danych wejściowych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# # One-hot encoding dla y_train i y_test
# y_train_encoded = to_categorical(y_train, num_classes=len(le.classes_))
# y_test_encoded = to_categorical(y_test, num_classes=len(le.classes_))

##############################

# # Definiowanie modelu sieci neuronowej
# model = Sequential()
# model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(len(le.classes_), activation='softmax'))

# # Kompilowanie modelu
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Trenowanie modelu
# model.fit(X_train_scaled, y_train_encoded, epochs=20, batch_size=64, validation_data=(X_test_scaled, y_test_encoded))

# # Ewaluacja modelu na zbiorze testowym
# y_pred_nn = model.predict(X_test_scaled)
# y_pred_nn_classes = y_pred_nn.argmax(axis=-1)

# # Ocena modelu
# accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)

# # Wyświetlenie wyników
# print("Neural Network Model Evaluation for 'attack_cat':")
# print(f"Accuracy: {accuracy_nn}")

# # Szczegółowy raport klasyfikacji
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_nn_classes, target_names=le.classes_))


######################################################

# Definiowanie modelu
def create_model(neurons=64, activation='relu', optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(le.classes_), activation='softmax'))  # Softmax dla klasyfikacji wieloklasowej
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Funkcja celu dla Optuna
def objective(trial):
    # Proponowanie wartości hiperparametrów przez Optuna
    n_layers = trial.suggest_int('n_layers', 1, 3)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    neurons = trial.suggest_int('neurons', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Budowanie modelu Keras
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    for i in range(n_layers):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(len(le.classes_), activation='softmax'))
    
    # Kompilowanie modelu
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Trenowanie modelu
    model.fit(X_train_scaled, y_train_encoded, epochs=20, batch_size=32, verbose=0)
    
    # Ewaluacja na zbiorze testowym
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Uzyskanie klasy z predykcji
    accuracy = accuracy_score(y_test_encoded, y_pred_classes)
    
    return accuracy

# Funkcja do zapisu wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

# Tworzenie i uruchamianie optymalizacji za pomocą Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Najlepsze parametry i wynik
best_params = study.best_params
best_value = study.best_value

save_results_to_file("nn_attack_cat_results.txt", f"Best parameters: {best_params}")
save_results_to_file("nn_attack_cat_results.txt", f"Best accuracy: {best_value}")


# Budowanie finalnego modelu z najlepszymi parametrami
best_trial = study.best_trial
final_model = Sequential()
final_model.add(Dense(best_trial.params['neurons'], input_dim=X_train_scaled.shape[1], activation=best_trial.params['activation']))
final_model.add(Dropout(best_trial.params['dropout_rate']))
for i in range(best_trial.params['n_layers']):
    final_model.add(Dense(best_trial.params['neurons'], activation=best_trial.params['activation']))
    final_model.add(Dropout(best_trial.params['dropout_rate']))
final_model.add(Dense(len(le.classes_), activation='softmax'))

final_model.compile(loss='sparse_categorical_crossentropy', optimizer=best_trial.params['optimizer'], metrics=['accuracy'])

# Trenowanie finalnego modelu
final_model.fit(X_train_scaled, y_train_encoded, epochs=50, batch_size=32, verbose=1)

# Ewaluacja finalnego modelu na zbiorze testowym
y_pred_final = final_model.predict(X_test_scaled)
y_pred_final_classes = np.argmax(y_pred_final, axis=1)
final_accuracy = accuracy_score(y_test_encoded, y_pred_final_classes)

# Zapis wyników do pliku tekstowego
save_results_to_file("nn_attack_cat_results.txt", f"Final Model Accuracy: {final_accuracy}")
save_results_to_file("nn_attack_cat_results.txt", "Classification Report:\n" + classification_report(y_test_encoded, y_pred_final_classes, target_names=le.classes_))

print(f"Final Model Accuracy: {final_accuracy}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_final_classes, target_names=le.classes_))




# # Funkcja do budowy modelu Keras z **kwargs
# def create_model(neurons=32, optimizer='adam', activation='relu', **kwargs):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
#     model.add(tf.keras.layers.Dense(neurons, activation=activation))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid dla klasyfikacji binarnej

#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

# # Funkcja do budowy modelu Keras z **kwargs
# def create_model(neurons=32, optimizer='adam', activation='relu', **kwargs):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
#     model.add(tf.keras.layers.Dense(neurons, activation=activation))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid dla klasyfikacji binarnej

#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

# # Użycie KerasClassifier
# model = KerasClassifier(build_fn=create_model, verbose=0)

# # Definiowanie przestrzeni przeszukiwań hiperparametrów
# param_dist = {
#     'batch_size': [32, 64, 128],
#     'epochs': [10, 20, 30],
#     'optimizer': ['adam', 'rmsprop', 'sgd'],
#     'activation': ['relu', 'tanh'],
#     'neurons': [32, 64, 128]
# }

# # RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
# random_search.fit(X_train_scaled, y_train)

# # Wyświetlenie najlepszych parametrów
# print(f"Best Parameters: {random_search.best_params_}")

# # Ocena najlepszego modelu na zbiorze testowym
# best_model = random_search.best_estimator_
# y_pred_nn = best_model.predict(X_test_scaled)
# y_pred_nn_classes = (y_pred_nn > 0.5).astype("int32")

# accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
# print("Optimized Neural Network Model Evaluation for 'label':")
# print(f"Accuracy: {accuracy_nn}")

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_nn_classes))

"""
"""






#### LABEL AS RESULTS ####

# Kopia danych treningowych
x_train = data_train.copy()
y_train = data_train['label']  # Zmiana na kolumnę 'label'

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

# StandardScaler i PowerTransformer
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
y_test = data_test['label']

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

# Przygotowanie danych wejściowych i wyjściowych
X_train = x_train.drop(columns=['label', 'attack_cat'])
y_train = x_train['label']
X_test = x_test.drop(columns=['label', 'attack_cat'])
y_test = x_test['label']

# Normalizacja danych wejściowych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definiowanie modelu
def create_model(neurons=64, activation='relu', optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid dla klasyfikacji binarnej
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Funkcja celu dla Optuna
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    neurons = trial.suggest_int('neurons', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Budowanie modelu Keras
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    for i in range(n_layers):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Zmiana na klasyfikację binarną
    
    # Kompilowanie modelu
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Trenowanie modelu
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Ewaluacja na zbiorze testowym
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = (y_pred > 0.5).astype("int32")  # Klasyfikacja binarna
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    return accuracy

# Funkcja do zapisu wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

# Tworzenie i uruchamianie optymalizacji za pomocą Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Najlepsze parametry i wynik
best_params = study.best_params
best_value = study.best_value

save_results_to_file("nn_label_results.txt", f"Best parameters: {best_params}")
save_results_to_file("nn_label_results.txt", f"Best accuracy: {best_value}")

# Budowanie finalnego modelu z najlepszymi parametrami
best_trial = study.best_trial
final_model = Sequential()
final_model.add(Dense(best_trial.params['neurons'], input_dim=X_train_scaled.shape[1], activation=best_trial.params['activation']))
final_model.add(Dropout(best_trial.params['dropout_rate']))
for i in range(best_trial.params['n_layers']):
    final_model.add(Dense(best_trial.params['neurons'], activation=best_trial.params['activation']))
    final_model.add(Dropout(best_trial.params['dropout_rate']))
final_model.add(Dense(1, activation='sigmoid'))  # Zmiana na klasyfikację binarną

final_model.compile(loss='binary_crossentropy', optimizer=best_trial.params['optimizer'], metrics=['accuracy'])

# Trenowanie finalnego modelu
final_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Ewaluacja finalnego modelu na zbiorze testowym
y_pred_final = final_model.predict(X_test_scaled)
y_pred_final_classes = (y_pred_final > 0.5).astype("int32")
final_accuracy = accuracy_score(y_test, y_pred_final_classes)

# Zapis wyników do pliku tekstowego
save_results_to_file("nn_label_results.txt", f"Final Model Accuracy: {final_accuracy}")
save_results_to_file("nn_label_results.txt", "Classification Report:\n" + classification_report(y_test, y_pred_final_classes))

print(f"Final Model Accuracy: {final_accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final_classes))