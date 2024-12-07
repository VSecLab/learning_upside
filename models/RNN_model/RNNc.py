import os
import pandas as pd

def create_mse_table(directory_path):
    # Lista di tutte le etichette per righe e colonne
    row_labels = []
    column_labels = []

    # Creazione della matrice vuota per i valori MSE
    mse_values = [[None] * 100 for _ in range(100)]

    # Iterazione su ogni file CSV
    for user_id in range(5):  # Utenti da 0 a 4
        for sequence_id in range(20):  # Sequenze da 0 a 19
            # Costruisci il nome del file CSV
            csv_filename = f"{user_id}_{sequence_id}_Head_Pitch_riconosci.csv"
            csv_path = os.path.join(directory_path, str(user_id), csv_filename)

            # Estrai l'MSE dalla colonna "mse" del file CSV
            mse_df = pd.read_csv(csv_path)
            mse_values[user_id * 20 + sequence_id] = mse_df['mse'].values

            # Aggiungi l'etichetta della riga
            row_labels.append(f"{user_id}_{sequence_id}")

            # Aggiungi l'etichetta della colonna
            column_labels.append(f"{user_id}_{sequence_id}")

    # Creazione del DataFrame Pandas
    mse_table_df = pd.DataFrame(mse_values, index=row_labels, columns=column_labels)

    # Salvataggio del DataFrame come file CSV
    mse_table_df.to_csv('mse_tableRNNauto2E_8B.csv')

# Esempio di utilizzo
create_mse_table("RNNauto2E_8B")


#RNN2E_8B
#0_0_['Head_Pitch', 'Head_Roll', 'Head_Yaw']_riconosci.csv