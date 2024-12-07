import os
import pandas as pd
import numpy as np



def calculate_threshold(directory_path):
    # Leggi i dati da tutti i file CSV nella directory e concatena i risultati in un unico DataFrame
    all_mse = []
    for user_id in range(5):
        user_directory = os.path.join(directory_path, str(user_id))
        for csv_filename in os.listdir(user_directory)[:10]:  # Prendi solo 10 file casuali
            csv_path = os.path.join(user_directory, csv_filename)
            mse_df = pd.read_csv(csv_path)
            all_mse.extend(mse_df['mse'])

    # Calcola il quartile desiderato (ad esempio, il 5%)
    threshold = pd.Series(all_mse).quantile(0.25)
    return threshold

# Esempio di utilizzo
directory_path = "conserva/RNNauto2E_8B"  # Cartella principale
threshold = calculate_threshold(directory_path)
print(f"Soglia calcolata utilizzando il quartile: {threshold}")


def count_accepted_mse(folder_path, threshold):
    num_accepted = 0

    # Itera su tutti i file CSV nella cartella
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            mse_values = df['mse'].tolist()

            # Conta quanti MSE sono al di sotto della soglia
            num_accepted += sum(1 for mse in mse_values if mse < threshold)

    return num_accepted

# Esempio di utilizzo
folder_path = "conserva/RNN2E_8B/0"
threshold = 0.002  # Sostituisci con la soglia calcolata
num_accepted = count_accepted_mse(folder_path, threshold)
print(f"Numero di MSE al di sotto della soglia: {num_accepted}")
