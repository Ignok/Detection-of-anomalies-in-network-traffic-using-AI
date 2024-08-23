import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

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

########################

# Funkcja do zapisu wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

# Testowanie różnych wartości encoding_dim
best_accuracy = 0
best_encoding_dim = None

for encoding_dim in [16, 32, 64, 128, 256]:
    print(f"Testing encoding_dim = {encoding_dim}")
    
    # Definiowanie autokodera
    input_dim = X_train_scaled.shape[1]

    # Model autokodera
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Kompilacja modelu
    autoencoder.compile(optimizer='adam', loss='mse')

    # Trenowanie autokodera
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_scaled, X_test_scaled), verbose=1)

    # Tworzenie modelu encoder-dekoder
    encoder_model = Model(inputs=input_layer, outputs=encoder)

    # Redukcja wymiarowości danych za pomocą wytrenowanego enkodera
    X_train_encoded = encoder_model.predict(X_train_scaled)
    X_test_encoded = encoder_model.predict(X_test_scaled)

    # Definiowanie modelu klasyfikacyjnego
    classifier = Sequential()
    classifier.add(Dense(32, input_dim=encoding_dim, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))  # Sigmoid dla klasyfikacji binarnej

    # Kompilowanie modelu klasyfikacyjnego
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Trenowanie modelu klasyfikacyjnego
    classifier.fit(X_train_encoded, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test_encoded, y_test_encoded), verbose=1)

    # Ewaluacja modelu na zbiorze testowym
    y_pred = classifier.predict(X_test_encoded)
    y_pred_classes = (y_pred > 0.5).astype("int32")  # Przekształcenie do klasyfikacji binarnej

    # Ocena modelu
    accuracy = accuracy_score(y_test_encoded, y_pred_classes)
    print(f"Accuracy for encoding_dim {encoding_dim}: {accuracy}")

    # Zapis wyników do pliku
    save_results_to_file("autoencoder_label_dim_results.txt", f"encoding_dim: {encoding_dim}, Accuracy: {accuracy}")

    # Aktualizacja najlepszego wyniku
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_encoding_dim = encoding_dim

# Zapis najlepszego wyniku
save_results_to_file("autoencoder_label_dim_results.txt", f"Best encoding_dim: {best_encoding_dim}, Best Accuracy: {best_accuracy}")

print(f"Best encoding_dim: {best_encoding_dim}, Best Accuracy: {best_accuracy}")


"""


# Definiowanie autokodera
input_dim = X_train_scaled.shape[1]
encoding_dim = 64  # Rozmiar warstwy ukrytej

# Model autokodera
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')

# Trenowanie autokodera
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_scaled, X_test_scaled), verbose=1)

# Tworzenie modelu encoder-dekoder
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Redukcja wymiarowości danych za pomocą wytrenowanego enkodera
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

# Definiowanie modelu klasyfikacyjnego
classifier = Sequential()
classifier.add(Dense(32, input_dim=encoding_dim, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))  # Sigmoid dla klasyfikacji binarnej

# Kompilowanie modelu klasyfikacyjnego
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trenowanie modelu klasyfikacyjnego
classifier.fit(X_train_encoded, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test_encoded, y_test_encoded), verbose=1)

# Ewaluacja modelu na zbiorze testowym
y_pred = classifier.predict(X_test_encoded)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Przekształcenie do klasyfikacji binarnej

# Ocena modelu
accuracy = accuracy_score(y_test_encoded, y_pred_classes)

# Zapis wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

save_results_to_file("autoencoder_label_results.txt", f"Final Model Accuracy: {accuracy}")
save_results_to_file("autoencoder_label_results.txt", "Classification Report:\n" + classification_report(y_test_encoded, y_pred_classes))

print(f"Final Model Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_classes))

"""

### Attack_cat as results ###
"""
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

# Definiowanie autokodera
input_dim = X_train_scaled.shape[1]
encoding_dim = 64  # Rozmiar warstwy ukrytej

# Model autokodera
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')

# Trenowanie autokodera
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_scaled, X_test_scaled), verbose=1)

# Tworzenie modelu encoder-dekoder
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Redukcja wymiarowości danych za pomocą wytrenowanego enkodera
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

# Definiowanie modelu klasyfikacyjnego
classifier = Sequential()
classifier.add(Dense(32, input_dim=encoding_dim, activation='relu'))
classifier.add(Dense(len(le.classes_), activation='softmax'))

# Kompilowanie modelu klasyfikacyjnego
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trenowanie modelu klasyfikacyjnego
classifier.fit(X_train_encoded, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test_encoded, y_test_encoded), verbose=1)

# Ewaluacja modelu na zbiorze testowym
y_pred = classifier.predict(X_test_encoded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ocena modelu
accuracy = accuracy_score(y_test_encoded, y_pred_classes)

# Zapis wyników do pliku tekstowego
def save_results_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + "\n")

save_results_to_file("autoencoder_attack_cat_results.txt", f"Final Model Accuracy: {accuracy}")
save_results_to_file("autoencoder_attack_cat_results.txt", "Classification Report:\n" + classification_report(y_test_encoded, y_pred_classes, target_names=le.classes_))

print(f"Final Model Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=le.classes_))

"""