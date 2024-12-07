import pandas as pd
import os

# Funzione per calcolare la media degli MSE per ogni utente in un DataFrame
def calcola_media_per_utente(df):
    return df.groupby('user')['mse'].mean().reset_index()

# Percorsi delle cartelle di input
input_folders = ['resultsMse_LSTM', 'resultsMse_LSTMlabel', 'resultsMse_RNN', 'resultsMse_RNNlabel']
#input_folders = ['conserva/lstmauto2E_8B', 'conserva/lstmauto2E_8Blabel3', 'conserva/RNN2E_8B', 'conserva/RNN2E_8Blabel3']

# Percorso della cartella di output
output_folder = 'media'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Esplora ciascuna cartella
for folder in input_folders:
    # Esplora ciascun file CSV nella cartella
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            # Carica il file CSV in un DataFrame pandas
            df = pd.read_csv(file_path)
            # Calcola la media degli MSE per ogni utente
            media_per_utente = calcola_media_per_utente(df)
            # Nome del file di output
            output_file_path = os.path.join(output_folder, f'{filename[:-4]}_media.csv')
            # Salva il risultato in un nuovo file CSV
            media_per_utente.to_csv(output_file_path, index=False)

print("Le medie sono state calcolate e salvate nella cartella 'media'.")
