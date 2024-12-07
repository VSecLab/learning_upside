import pandas as pd
import numpy as np
import VisorData as vd
import time as tm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
import os


def create_model(df):
    epochs = 2
    batch_size = 8

    # Rimuovi le righe con valori mancanti
    df = df.dropna()

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Reshape per adattarsi all'input del modello RNN (necessario per la forma [samples, timesteps, features])
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    # Configurazione dell'architettura del modello RNN autoencoder
    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(df_scaled.shape[1], 1)))
    model.add(RepeatVector(df_scaled.shape[1]))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(TimeDistributed(Dense(units=1)))
    model.compile(optimizer='adam', loss='mse')

    # Addestramento del modello
    model.fit(df_scaled, df_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return model, scaler

def evaluate_model(lstm, scaler, df):
    # Predizione sui dati di addestramento
    df = df.dropna()

    # Normalizzazione dei dati
    df_scaled = scaler.transform(df)

    # Reshape per adattarsi all'input del modello LSTM (necessario per la forma [samples, timesteps, features])
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    predictions_scaled = lstm.predict(df_scaled)

    
    # Calcolo dell'errore di ricostruzione (Mean Squared Error)
    mse = np.mean(np.power(df_scaled - predictions_scaled, 2))
    print(f'Mean Squared Error: {mse}')

    # # Denormalizzazione delle predizioni
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, df_scaled.shape[1]))
    # df_unscaled = scaler.inverse_transform(df_scaled.reshape(-1, df_scaled.shape[1]))
    # print('predictions')
    # print(predictions)
    # plot(df_unscaled,predictions)
    return predictions,mse


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

    filename = f'RNNauto2E_8B/{userid}/{userid}_{sid}_{parameter}_riconosci.csv'
    df.insert(2, "check", riconosco, True)
    df.to_csv(filename, index=False)  # Utilizzo di index=False per evitare di salvare l'indice nel file CSV
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
    dfs=vd.get_all_df_for_user(data,users[userid],'Head_Pitch')
    df=dfs[seqIDs[seqid]]  

    print("create model")
    tm.sleep(1)
    lstm, scaler=create_model(df)
    print("evaluate model")
    tm.sleep(1)
    predict=evaluate_model(lstm,scaler, df)
    #print(predict)



def model_and_evaluateRNNauto(userid, sid, parameter):
    data = vd.get_all_users_()
    users = vd.get_users()

    # Creare il modello per uno degli utenti e una delle sequenze
    username = users[userid]
    seqIDs = vd.get_SeqIDs_user(data, users[userid])
    dfs = vd.get_all_df_for_user(data, users[userid], parameter)
    df = dfs[seqIDs[sid]]
    rnn_model, scaler = create_model(df)

    # Valutare l'errore quadratico medio (MSE) per tutte le sequenze
    schema = {'user': [], 'mse': []}
    mses = pd.DataFrame(schema)
    
    for user in users:
        print('Valutazione utente:' + user)
        seqIDs = vd.get_SeqIDs_user(data, user)
        dfs = vd.get_all_df_for_user(data, user, parameter)
        
        for seqid in seqIDs:
            df = dfs[seqid]
            print('Valutazione utente:' + user + ' sulla sequenza:' + seqid)
            tm.sleep(1)
            predictions, mse = evaluate_model(rnn_model, scaler, df)
            print(mse)
            mses.loc[len(mses)] = [user, mse]
        print('Prossimo utente')

    mses.to_csv('results/mses' + username + '_' + str(sid) + '.csv')



# Aggiungi una funzione per verificare l'accuratezza con RNN
def check_rnnauto(userid, sid, parameter,threshold):
    data = vd.get_all_users_()
    users = vd.get_users()

    # Creare il modello per uno degli utenti e una delle sequenze
    username = users[userid]
    df = pd.read_csv('results/mses' + username + '_' + str(sid) + '.csv')
    df2 = riconosci(df,userid,sid,parameter,threshold)
    tp, fp, tn, fn = conta(df2, username)
    precision_value = precision(tp, fp, tn, fn)
    recall_value = recall(tp, fp, tn, fn)
    results_df = pd.DataFrame({'User_ID': [userid], 'Seq_ID': [sid], 'Precision': [precision_value], 'Recall': [recall_value]})
    results_df.to_csv('RNNauto28precision_recall_results.csv', mode='a', index=False, header=not os.path.exists('RNNauto28precision_recall_results.csv'))
    print('precisione:' + str(precision(tp, fp, tn, fn)))
    print('recall:' + str(recall(tp, fp, tn, fn)))
    print('tp,  fp, tn, fn')
    print(tp, fp, tn, fn)
    plotMSEs(df, str(sid))
    cm = np.array([[tn, fp], [fn, tp]])  # Costruisci la matrice di confusione direttamente
    print('Matrice di confusione:')
    print(cm)
    
    # Salva la matrice di confusione in un file
    filename = f'matriceRNNauto2_8/{username}_{sid}_{parameter}_{threshold}_confusion_matrix.npy'
    np.save(filename, cm)


def run_experiments(parameter, threshold):
    data = vd.get_all_users_()
    users = vd.get_users()

    for userid in range(5):   # 5 userid da 0 a 4
     for sid in range(20):  # 20 sid da 0 a 19
            # Puoi cambiare il parametro se necessario
            print(f"Running experiments for userid {userid}, sid {sid}")

            # Esegui il training del modello e valutazione
            model_and_evaluateRNNauto(userid, sid, parameter)

            # Esegui la verifica
            check_rnnauto(userid, sid, parameter, threshold)

# Esegui gli esperimenti
run_experiments('Head_Pitch', 0.05)


# model_and_evaluate (userid, sid, parameter):
#model_and_evaluateRNNauto(0, 1, 'Head_Pitch')
    
# Esegui la verifica con RNN
#check_rnnauto(0, 1, 0.05)