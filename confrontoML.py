import os
import pandas as pd
import matplotlib.pyplot as plt

def confronto_grafico(file1_path, file2_path, nome_interessato, nome_etichetta1, nome_etichetta2):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Grafico 1: Confronto tra "si" e "no" per il nome inserito
    fig, axes = plt.subplots(nrows=2, figsize=(10, 12))

    df1_user = df1[df1['user'] == nome_interessato]
    df2_user = df2[df2['user'] == nome_interessato]

    axes[0].bar([f'Si ({nome_etichetta1})', f'No ({nome_etichetta1})', f'Si ({nome_etichetta2})', f'No ({nome_etichetta2})'], [
            len(df1_user[df1_user['check'] == 'si']),
            len(df1_user[df1_user['check'] == 'no']),
            len(df2_user[df2_user['check'] == 'si']),
            len(df2_user[df2_user['check'] == 'no'])])

    axes[0].set_title(f'Confronto tra "si" e "no" per {nome_interessato}')
    axes[0].set_ylabel('Frequenza')

    # Grafico 2: Quante volte ogni altra persona ha sbagliato
    df1['errore'] = (df1['check'] == 'si').astype(int)
    df2['errore'] = (df2['check'] == 'si').astype(int)

    errore_per_persona = pd.DataFrame({
        f'Errore ({nome_etichetta1})': df1.groupby('user')['errore'].sum(),
        f'Errore ({nome_etichetta2})': df2.groupby('user')['errore'].sum()
    })

    errore_per_persona = errore_per_persona.reset_index()
    
    errore_per_persona[errore_per_persona['user'] != nome_interessato].plot(kind='bar', x='user', y=[f'Errore ({nome_etichetta1})', f'Errore ({nome_etichetta2})'], ax=axes[1])

    axes[1].set_title(f'Quante volte ogni altra persona ha sbagliato ("si" al posto di "no")')
    axes[1].set_xlabel('Nome Utente')
    axes[1].set_ylabel('Frequenza di errore')
    axes[1].legend([f'Errore ({nome_etichetta1})', f'Errore ({nome_etichetta2})'])

    plt.savefig(os.path.join('plots', 'confronto_alelstmAuto32_rnn.png'))
    plt.show()




confronto_grafico("/Users/alessandromercurio/Downloads/learningVR/lstm5epoche/conte/0_12_Head_Pitch_riconosci.csv", "/Users/alessandromercurio/Downloads/learningVR/RNN5label3/conte/0_5_['Head_Pitch', 'Head_Roll', 'Head_Yaw']_riconosci.csv", 'ConteTeresa', 'LSTMAutoencoderlabel3', 'RNNlabel3')
