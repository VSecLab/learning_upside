import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dati combinati dai 5 CSV
percorso_cartella_unione = 'unione'
frames = []
for csv_file in os.listdir(percorso_cartella_unione):
    if csv_file.endswith('_unione.csv'):
        percorso_csv = os.path.join(percorso_cartella_unione, csv_file)
        df = pd.read_csv(percorso_csv)
        # Rimuovi le colonne 'Timestamp' e 'Persona', se presenti
        if 'Timestamp' in df.columns:
            df.drop(columns=['Timestamp'], inplace=True)
        if 'Persona' in df.columns:
            df.drop(columns=['Persona'], inplace=True)
        frames.append(df)

df_unione = pd.concat(frames, ignore_index=True)

# Utilizza solo la feature "Head_Pitch"
X = df_unione['Head_Pitch'].values.reshape(-1, 1)

# Codifica della variabile target 'Utente'
label_encoder = LabelEncoder()
df_unione['Utente'] = label_encoder.fit_transform(df_unione['Utente'])
y = df_unione['Utente'].values

# Standardizzazione delle features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape dei dati per l'input al modello LSTM (3D array)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Creazione del modello LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestramento del modello
history = model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=2, validation_data=(X_test, y_test))

# Valutazione del modello sul test set
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))

# Predizione sul test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calcolo della matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matrice di confusione:')
print(conf_matrix)

# Plot della loss durante l'addestramento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot dell'andamento dell'accuratezza durante l'addestramento
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot della matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matrice di Confusione')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()
