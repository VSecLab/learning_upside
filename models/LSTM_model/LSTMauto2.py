import pandas as pd
import numpy as np
import VisorData as vd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt

epochs = 10
batch_size = 32

data = vd.get_all_users_()
users=vd.get_users()

epochs = 10
batch_size = 32

#df = pd.read_csv('orientationData_1_26590c7e-8b65-44c7-bed9-bb4dc4f25ef6.csv')

# Seleziona solo le colonne di interesse (Timestamp, Head_Pitch)
#df = df[['Timestamp', 'Head_Pitch']]
userid=3
testid=1
df=data.loc[data['user'] == users[userid]]
pins=vd.get_pins(df)
seqIDs=vd.get_SeqIDs(df)

tss={}
for id in seqIDs:
    tss[id]=vd.get_df(data,id,'Head_Pitch')

df=tss[seqIDs[2]]  

testdf=data.loc[data['user'] == users[testid]]
testdfpins=vd.get_pins(testdf)
testdfseqIDs=vd.get_SeqIDs(testdf)

tss={}
for id in testdfseqIDs:
    tss[id]=vd.get_df(data,id,'Head_Pitch')

df_new=tss[testdfseqIDs[2]]  



# Rimuovi le righe con valori mancanti
df = df.dropna()

# Normalizzazione dei dati
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Reshape per adattarsi all'input del modello LSTM (necessario per la forma [samples, timesteps, features])
df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

# Configurazione dell'architettura del modello LSTM autoencoder
model = Sequential()
model.add(LSTM(units=64, input_shape=(df_scaled.shape[1], 1)))
model.add(RepeatVector(df_scaled.shape[1]))
model.add(LSTM(units=64, return_sequences=True))
model.add(TimeDistributed(Dense(units=1)))
model.compile(optimizer='adam', loss='mse')

# Addestramento del modello
model.fit(df_scaled, df_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Predizione sui dati di addestramento
predictions_scaled = model.predict(df_scaled)

# Denormalizzazione delle predizioni
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, df_scaled.shape[1]))

# Calcolo dell'errore di ricostruzione (Mean Squared Error)
mse = np.mean(np.power(df_scaled - predictions_scaled, 2))
print(f'Mean Squared Error: {mse}')

# nuovo CSV

#df_new = tss[seqIDs[5]]  


# Rimuovi le righe con valori mancanti
df_new = df_new.dropna()

# Normalizzazione dei dati
df_new_scaled = scaler.transform(df_new)

# Reshape per adattarsi all'input del modello LSTM (necessario per la forma [samples, timesteps, features])
df_new_scaled = df_new_scaled.reshape((df_new_scaled.shape[0], df_new_scaled.shape[1], 1))

# Predizione sui dati del nuovo file
predictions_new_scaled = model.predict(df_new_scaled)

# Denormalizzazione delle predizioni
predictions_new = scaler.inverse_transform(predictions_new_scaled.reshape(-1, df_new_scaled.shape[1]))

# Calcolo dell'errore di ricostruzione (Mean Squared Error)
mse_new = np.mean(np.power(df_new_scaled - predictions_new_scaled, 2))
print(f'Mean Squared Error on New Data: {mse_new}')


# Denormalizzazione dei dati reali
df_new_unscaled = scaler.inverse_transform(df_new_scaled.reshape(-1, df_new_scaled.shape[1]))

# Denormalizzazione delle previsioni
predictions_new_unscaled = scaler.inverse_transform(predictions_new_scaled.reshape(-1, df_new_scaled.shape[1]))

# Estrai colonne specifiche per il grafico (Timestamp e Head_Pitch)
timestamp = df_new_unscaled[:, 0]
head_pitch_real = df_new_unscaled[:, 1]
head_pitch_pred = predictions_new_unscaled[:, 1]

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(timestamp, head_pitch_real, label='Dati reali', marker='o')
plt.plot(timestamp, head_pitch_pred, label='Previsioni', marker='x')
plt.xlabel('Timestamp')
plt.ylabel('Head_Pitch')
plt.title('Confronto Dati '+users[userid]+' previsioni di'+users[testid])
plt.legend()
#plt.show()
plt.savefig('plots/compare.png')
