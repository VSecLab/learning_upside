import os
import random 
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.layers import SimpleRNN, Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error

def set_seed():
    random.seed(50)
    np.random.seed(50)
    tf.random.set_seed(50)

def create_sequence(df, timesteps):
    """
    Create sequences of data for the model

    """

    # prende in ingresso un array di array 
    # ogni array è una riga del csv 
    # ogni riga è un array di 3 elementi
    # mi restituisce un array 3-D
    # l'array più interno è un array di 3 elementi ([x, y, z])
    # ogni array intermedio è un blocco di timesteps ([[x1, y1, z1], [x2, y2, z2], ...])
    # l'array esterno è l'insieme di tutti i blocchi ([[[x11, y11, z11], [x12, y12, z12], ...], [[x21, y21, z21], [x22, y22, z22], ...], ...])
    sequences = []
    targets = []
    for i in range(len(df) - timesteps):
        sequences.append(df[i:i + timesteps])
        targets.append(df[i + timesteps])
    return np.array(sequences), np.array(targets)

def create_sequence_autoencoder(df, timesteps):
    sequences = []
    for i in range(len(df) - timesteps):
        sequences.append(df[i:i + timesteps])
    return np.array(sequences)

def create_model(df, epochs, batch_size, model_type):

    timesteps = 100
    
    df = df.dropna()
    if df.empty:
        raise ValueError("\nThe input DataFrame is empty after dropping missing values.")

    # Reshape to fit the model input (necessary for the shape [samples, timesteps, features])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='loss', 
        patience=3, 
        mode = 'min',
        restore_best_weights=True
    )

    model = Sequential()

    if model_type == 'LSTMlabel':
        x_train, y_train = create_sequence(df_scaled, timesteps)
        # shape[0] -> numero di entry totali; shape[1] -> numero di timesteps in ogni squenza (100); shape[2] -> numero di features (3)
        # print(f"\ncreate_model() - x_train.shape[0]: {x_train.shape[0]} - x_train.shape[1]: {x_train.shape[1]} - x_train.shape[2]: {x_train.shape[2]}\n")
        
        model.add(LSTM(units=50, input_shape=(x_train.shape[1], 3), return_sequences=True))
        model.add(LSTM(units=50, input_shape=(x_train.shape[1], 3), return_sequences=False))
        model.add(Dense(units=3)) 
        
        print("------LSTM model------\n")
    elif model_type == 'LSTMauto': 
        x_train = create_sequence_autoencoder(df_scaled, timesteps)

        model.add(LSTM(units=50, input_shape=(x_train.shape[1], 3), return_sequences=False))
        model.add(RepeatVector(x_train.shape[1]))
        model.add(LSTM(units=50, input_shape=(x_train.shape[1], 3), return_sequences=True))
        model.add(TimeDistributed(Dense(units=3))) 
        
        print("------LSTM model (autoencoder)------\n")
    else:
        print(f"\nModel type '{model_type}' is not supported.")
        raise ValueError("Invalid model_type. Supported types are 'LSTMauto', 'LSTMlabel'.\n")

    model.compile(optimizer='adam', loss='mse')
    
    if model_type == 'LSTMlabel':
        # Model training
        history = model.fit(
            x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=[early_stopping],
            verbose=1
        )

        # calcolo della mse e std sui dati di train 
        k = 3 # iperparametro: mi dice quanto sono conservativo nel definire il threshold

        y_pred = model.predict(x_train)
        mse = mean_squared_error(y_train, y_pred)
        std = np.std(mse)
        mse_tmp = (y_train - y_pred)**2 
        std_tmp = np.std(mse_tmp)
        t = mse_tmp + k*std_tmp
    
        print("\ncreate_model() - Training MSE:")
        print(mse_tmp)
        print("\ncreate_model() - Training STD:")
        print(std_tmp)
        print("\ncreate_model() - Threshold tmp:")
        print(t)
        print("\ncreate_model() - Training MSE:")
        print(mse)
        print("\ncreate_model() - Training STD:")
        print(std)
        threshold_tmp = mse + k*std 
        print("\ncreate_model() - Threshold:")
        print(threshold_tmp)

    elif model_type == 'LSTMauto': 
        # Model training
        history = model.fit(
            x_train, x_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=[early_stopping],
            verbose=1
        )

    training_loss = history.history['loss']
    training_loss = training_loss[-1]
    
    print("\ncreate_model() - Training loss:")
    print(training_loss)
    
    return model, scaler, training_loss