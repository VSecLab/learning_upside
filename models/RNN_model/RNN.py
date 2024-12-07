import pandas as pd
import numpy as np
import VisorData as vd
import time as tm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import os


def create_modelRNN(df, epochs, batch_size):
   

    # Rimuovi le righe con valori mancanti
    df = df.dropna()

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Reshape per adattarsi all'input del modello RNN (necessario per la forma [samples, timesteps, features])
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    # Configurazione dell'architettura del modello RNN
    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(df_scaled.shape[1], 1)))
    model.add(Dense(units=df_scaled.shape[1]))  # Output per ogni passo temporale
    model.compile(optimizer='adam', loss='mse')
    # Addestramento del modello
    model.fit(df_scaled, df_scaled, epochs, batch_size, validation_split=0.1)
    return model, scaler


def plot(df1, df2):
    # Estrai colonne specifiche per il grafico (Timestamp e Head_Pitch)
    timestamp = df1[:, 0]
    head_pitch_real = df1[:, 1]
    head_pitch_pred = df2[:, 1]

    # Crea il grafico
    plt.figure(figsize=(12, 6))
    plt.plot(timestamp, head_pitch_real, label='Dati reali', marker='o')
    plt.plot(timestamp, head_pitch_pred, label='Previsioni', marker='x')
    plt.xlabel('Timestamp')
    plt.ylabel('Head_Pitch')
    plt.legend()
    plt.show()

def plotMSEs(df, id):
    l = len(df)
    users = pd.unique(df['user'])
    for user in users:
        print('plot ' + user)
        tm.sleep(1)
        plt.clf()
        plt.xlabel(user)
        plt.xlabel('MSE')
        df2 = df.loc[df['user'] == user]
        plt.plot(df2['mse'])
        plt.savefig('plots/RNN_{parameter}_mses_{threshold}_{sid}' + user + '_' + id + '.png')

def riconosci(df, userid, sid, parameter, threshold):
    riconosco = []

    for index, row in df.iterrows():
        if row['mse'] < threshold:
            riconosco.append('si')
        else:
            riconosco.append('no')

    # Definire il percorso della cartella principale e della sotto-cartella
    directory = f'RNNcheck/{userid}_{parameter}_{threshold}'
    os.makedirs(directory, exist_ok=True)

# Definire il percorso completo del file CSV all'interno della sotto-cartella
    filename = f'{directory}/{userid}_{sid}_{parameter}_riconosci.csv'

# Insert, salvataggio del DataFrame come file CSV come prima
    df.insert(2, "check", riconosco, True)
    df.to_csv(filename, index=False)
 # Utilizzo di index=False per evitare di salvare l'indice nel file CSV
    return df

def conta(df, username):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index, row in df.iterrows():
        if row['user'] == username and row['check'] == 'si':
            tp = tp + 1
        elif row['user'] == username and row['check'] == 'no':
            fn = fn + 1
        elif row['user'] != username and row['check'] == 'si':
            fp = fp + 1
        elif row['user'] != username and row['check'] == 'no':
            tn = tn + 1
        else:
            print('sei pazzo')
    return tp, fp, tn, fn

def precision(tp, fp, tn, fn):
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def recall(tp, fp, tn, fn):
    
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)
    

def model_and_evaluate_single():
    data = vd.get_all_users_()
    users = vd.get_users()

    userid = 0
    seqid = 1
    seqIDs = vd.get_SeqIDs_user(data, users[userid])
    dfs = vd.get_all_df_for_user(data, users[userid], 'Head_Pitch')
    df = dfs[seqIDs[seqid]]

    print("create model")
    tm.sleep(1)
    rnn_model, scaler = create_modelRNN(df)
    print("evaluate model")
    tm.sleep(1)
    evaluate_model(rnn_model, scaler, df)

def model_and_evaluateRNN3label(userid, sid, features, epochs, batch_size):
    data = vd.get_all_users_()
    users = vd.get_users()

    # Creare il modello per uno degli utenti e una delle sequenze
    username = users[userid]
    seqIDs = vd.get_SeqIDs_user(data, users[userid])
    dfs = vd.get_all_df_for_user_with_timestamp(data, users[userid], features)
    df = dfs[seqIDs[sid]]
    rnn_model, scaler = create_modelRNN(df, epochs, batch_size)

    # Valutare l'errore quadratico medio (MSE) per tutte le sequenze
    schema = {'user': [], 'mse': []}
    mses = pd.DataFrame(schema)
    
    for user in users:
        print('Valutazione utente:' + user)
        seqIDs = vd.get_SeqIDs_user(data, user)
        dfs = vd.get_all_df_for_user_with_timestamp(data, user, features)
        
        for seqid in seqIDs:
            df = dfs[seqid]
            print('Valutazione utente:' + user + ' sulla sequenza:' + seqid)
            tm.sleep(1)
            mse = evaluate_model(rnn_model, scaler, df)
            print(mse)
            mses.loc[len(mses)] = [user, mse]
        print('Prossimo utente')

    mses.to_csv('resultsMseRNN/RNN_{features}_{epochs}_{batch_size}_mse' + username + '_' + str(sid) + '.csv')

def evaluate_model(rnn_model, scaler, df):
    # Predizione sui dati di addestramento
    df = df.dropna()

    # Normalizzazione dei dati
    df_scaled = scaler.transform(df)

    # Reshape per adattarsi all'input del modello RNN
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    predictions_scaled = rnn_model.predict(df_scaled)

    # Calcolo dell'errore di ricostruzione (Mean Squared Error)
    mse = np.mean(np.power(df_scaled - predictions_scaled[:, :, np.newaxis], 2))

    print(f'Mean Squared Error: {mse}')

    return mse

def check_rnn3label(userid, sid, parameter, threshold, epochs, batch_size):
    data = vd.get_all_users_()
    users = vd.get_users()

    # Creare il modello per uno degli utenti e una delle sequenze
    username = users[userid]
    df = pd.read_csv('resultsMseRNN/RNN_{features}_{epochs}_{batch_size}_mse' + username + '_' + str(sid) + '.csv')
    df2 = riconosci(df, userid, sid, parameter, threshold)
    tp, fp, tn, fn = conta(df2, username)
    precision_value = precision(tp, fp, tn, fn)
    recall_value = recall(tp, fp, tn, fn)
    results_df = pd.DataFrame({'User_ID': [userid], 'Seq_ID': [sid], 'Precision': [precision_value], 'Recall': [recall_value]})
    results_df.to_csv('RNNprecision_recall/RNN{userid}}_{sid}_{parameter}_{threshold}_{epochs}_{batch_size}_precision_recall_results.csv', mode='a', index=False, header=not os.path.exists('RNNprecision_recall/RNN{userid}}_{sid}_{parameter}_{threshold}_{epochs}_{batch_size}_precision_recall_results.csv'))
    print('precisione:' + str(precision(tp, fp, tn, fn)))
    print('recall:' + str(recall(tp, fp, tn, fn)))
    print('tp,  fp, tn, fn')
    print(tp, fp, tn, fn)
    plotMSEs(df, str(sid))
    # Calcola la matrice di confusione
    cm = np.array([[tn, fp], [fn, tp]])  # Costruisci la matrice di confusione direttamente
    print('Matrice di confusione:')
    print(cm)
    
    # Salva la matrice di confusione in un file
    filename = f'matriceRNN/{username}_{sid}_{parameter}_{threshold}_confusion_matrix.npy'
    np.save(filename, cm)

def run_experiments(parameter, threshold, epochs, batch_size):
    data = vd.get_all_users_()
    users = vd.get_users()

    for userid in range(5):   # 5 userid da 0 a 4
     for sid in range(20):  # 20 sid da 0 a 19
            # Puoi cambiare il parametro se necessario
            print(f"Running experiments for userid {userid}, sid {sid}")

            # Esegui il training del modello e valutazione
            model_and_evaluateRNN3label(userid, sid, parameter, epochs, batch_size)

            # Esegui la verifica
            check_rnn3label(userid, sid, parameter, threshold, epochs, batch_size)

# Esegui gli esperimenti
#run_experiments('Head_Pitch',0.05)


# Esempio di utilizzo
model_and_evaluateRNN3label(0, 19, 'Head_Pitch',2,8)
# Esegui la verifica con RNN
#check_rnn3label(0, 19,'Head_Pitch',0.05)
#if __name__ == "__main__":
    # Il codice qui verrà eseguito solo quando il modulo viene eseguito come script principale
   # pass
