import pandas as pd
import numpy as np
import VisorData as vd
import time as tm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def create_modelLSTMlabel(df, epochs, batch_size):
   

    # Rimuovi le righe con valori mancanti
    df = df.dropna()

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Reshape per adattarsi all'input del modello LSTM autoencoder
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    # Configurazione dell'architettura del modello LSTM autoencoder
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(df_scaled.shape[1], 1)))
    model.add(RepeatVector(df_scaled.shape[1]))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(TimeDistributed(Dense(units=1)))
    model.compile(optimizer='adam', loss='mse')

    # Addestramento del modello
    model.fit(df_scaled, df_scaled, epochs, batch_size, validation_split=0.1)
    
    return model, scaler



def plot(df1,df2):
    # Estrai colonne specifiche per il grafico (Timestamp e Head_Pitch)
    # timestamp = df1[:, 0]
    # head_pitch_real = df1[:, 1]
    # head_pitch_pred = df2[:, 1]

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
    #plt.title('Confronto Dati '+users[userid]+' previsioni di'+users[testid])
    plt.legend()
    plt.show()
    #plt.savefig('plots/compare.png')


def plotMSEs(df,id):
        l=len(df)
        users=pd.unique(df['user'])
        for user in users:
            print('plot '+user)
            tm.sleep(1)
            plt.clf()
            plt.xlabel(user)
            plt.xlabel('MSE')
            df2=df.loc[df['user'] == user]
            #plt.axis([0, 25, 0, 0.20])
            plt.plot(df2['mse'])
            #print(df2['mse'])
            plt.savefig('plots/mses'+user+'_'+id+'.png')

def riconosci(df, userid, sid, parameter, threshold):
    riconosco = []

    for index, row in df.iterrows():
        if row['mse'] < threshold:
            riconosco.append('si')
        else:
            riconosco.append('no')

    # Definire il percorso della cartella principale e della sotto-cartella
    directory = f'LSTMautocheck/{userid}_{parameter}_{threshold}'
    os.makedirs(directory, exist_ok=True)

# Definire il percorso completo del file CSV all'interno della sotto-cartella
    filename = f'{directory}/{userid}_{sid}_{parameter}_riconosci.csv'

# Insert, salvataggio del DataFrame come file CSV come prima
    df.insert(2, "check", riconosco, True)
    df.to_csv(filename, index=False)
 # Utilizzo di index=False per evitare di salvare l'indice nel file CSV
    return df

def conta(df,username):
    tp=0
    fp=0
    tn=0
    fn=0
    for index, row in df.iterrows():
        #print(row['user']+' '+row['check'])
        if row['user']==username and row['check']=='si':            
            tp=tp+1
        elif row['user']==username and row['check']=='no':
            fn=fn+1        
        elif row['user']!=username and row['check']=='si':
            fp=fp+1
        elif row['user']!=username and row['check']=='no':
            tn=tn+1
        else:
            print('sei pazzo')
        #print([tp,fp,tn,fn])
    return tp,fp,tn,fn

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
    users=vd.get_users()

    userid=0
    seqid=1
    seqIDs=vd.get_SeqIDs_user(data,users[userid])
    dfs=vd.get_all_df_for_user(data,users[userid],'Head_Yaw')
    df=dfs[seqIDs[seqid]]  

    print("create model")
    tm.sleep(1)
    lstm, scaler=create_modelLSTMlabel(df)
    print("evaluate model")
    tm.sleep(1)
    predict=evaluate_lstm_autoencoder(lstm,scaler, df)
    #print(predict)


def evaluate_lstm_autoencoder(lstm_autoencoder_model, df):
    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Predizione sui dati di addestramento
    predictions_scaled = lstm_autoencoder_model.predict(df_scaled)
    
    # Calcolo dell'errore di ricostruzione (Mean Squared Error)
    mse = np.mean(np.power(df_scaled.squeeze() - predictions_scaled.squeeze(), 2))

    print(f'Mean Squared Error: {mse}')

    return mse








def model_and_evaluate_lstm_autoencoder(userid, sid, features, epochs, batch_size):
    data = vd.get_all_users_()
    users = vd.get_users()

    # Creare il modello per uno degli utenti e una delle sequenze
    username = users[userid]
    seqIDs = vd.get_SeqIDs_user(data, users[userid])
    dfs = vd.get_all_df_for_user_with_timestamp(data, users[userid], features)
    df = dfs[seqIDs[sid]]
    
    lstm_autoencoder_model, scaler = create_modelLSTMlabel(df, epochs, batch_size)

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
            mse = evaluate_lstm_autoencoder(lstm_autoencoder_model,df)
            print(mse)
            mses.loc[len(mses)] = [user, mse]
        print('Prossimo utente')

    mses.to_csv('resultsMseLSTMauto/RNN3label_{features}_{epochs}_{batch_size}_mse' + username + '_' + str(sid) + '.csv')

# Esegui la verifica con LSTM autoencoder



def CheckLSTM3label(userid,seqid,parameter,threshold):
    data = vd.get_all_users_()
    users=vd.get_users()
    #Create the model for one of the user and one of the sequences
    username=users[userid]
    df=pd.read_csv('results/mses'+username+'_'+str(seqid)+'.csv')
    df2=riconosci(df,userid,seqid,parameter,threshold)
    tp, fp, tn, fn=conta(df2,username)
    print('precision:'+str(precision(tp,fp,tn,fn)))
    print('recall:'+str(recall(tp,fp,tn,fn)))
    print('tp,  fp, tn, fn')
    print(tp,fp,tn,fn)
    plotMSEs(df,str(seqid))

def run_experiments(userid, parameter, threshold):
    data = vd.get_all_users_()
    users = vd.get_users()

      # 5 userid da 0 a 4
    for sid in range(20):  # 20 sid da 0 a 19
            # Puoi cambiare il parametro se necessario
            print(f"Running experiments for userid {userid}, sid {sid}")

            # Esegui il training del modello e valutazione
            model_and_evaluate_lstm_autoencoder(userid, sid, parameter)

            # Esegui la verifica
            CheckLSTM3label(userid, sid, parameter, threshold)

# Esegui gli esperimenti
#run_experiments(0, ['Head_Pitch', 'Head_Roll', 'Head_Yaw'], 0.05)




# Esegui la verifica con LSTM autoencoder
#model_and_evaluate_lstm_autoencoder(0, 2, ['Head_Pitch', 'Head_Roll', 'Head_Yaw'])
#CheckLSTM3label(0,2,['Head_Pitch', 'Head_Roll', 'Head_Yaw'],0.05)
            
if __name__ == "__main__":
    # Il codice qui verrà eseguito solo quando il modulo viene eseguito come script principale
    pass
