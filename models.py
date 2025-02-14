from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, RepeatVector, TimeDistributed
import random 
import numpy as np
import tensorflow as tf
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

def create_model(df, epochs, batch_size, model_type):
    #set_seed()
    # Remove rows with missing values
    df = df.dropna()

    # Check if the DataFrame is empty after dropping NA
    if df.empty:
        raise ValueError("The input DataFrame is empty after dropping missing values.")

    # Reshape to fit the model input (necessary for the shape [samples, timesteps, features])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    """print("create_model() - df_scaled:")
    print(df_scaled)"""

    # Reshape to fit the model input (necessary for the shape [samples, timesteps, features])
    # df_scaled.shape[0] = number of samples
    # df_scaled.shape[1] = number of timesteps 
    # 1 = number of features

    # numero blocchi 
    # numero elemeneti per blocco 
    # numero di features per blocco

    timesteps = 100

    x_train, y_train = create_sequence(df_scaled, timesteps)

    
    print("create_model() - x_train:")
    print(x_train)

    print("create_model() - x_train.shape[0]:")
    print(x_train.shape[0])

    print("create_model() - x_train.shape[1]:")
    print(x_train.shape[1])

    print("create_model() - x_train.shape[2]:")
    print(x_train.shape[2])

    print("create_model() - y_train[-1] - x_train[-1]")
    print(f"{y_train[0]} - {x_train[0]}")

    #df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))
    
    # Model architecture configuration
    model = Sequential()

    if model_type == 'RNN' or model_type == 'RNN3label':
        model.add(SimpleRNN(units=32, input_shape=(df_scaled.shape[1], 3)))
        model.add(Dense(units=df_scaled.shape[1]))  # Output per ogni passo temporale
        print("------RNN model------")
    elif model_type == 'LSTMauto' or model_type == 'LSTMlabel':
        # model.add(LSTM(units=32, input_shape=(df_scaled.shape[1], 3)))
        model.add(LSTM(units=32, input_shape=(x_train.shape[1], 3), return_sequences=True))
        # model.add(RepeatVector(x_train.shape[1]))
        model.add(LSTM(units=32, input_shape=(x_train.shape[1], 3), return_sequences=False))
        # model.add(TimeDistributed(Dense(units=1)))
        model.add(Dense(units=3))
        
        print("------LSTMauto model------")
    else:
        print(f"Model type '{model_type}' is not supported.")
        raise ValueError("Invalid model_type. Supported types are 'RNN', 'RNN3label', 'LSTMauto', 'LSTMlabel'.")

    model.compile(optimizer='adam', loss='mse')
    
    # Model training
    #history = model.fit(df_scaled, df_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    training_loss = history.history['loss']
    training_loss = training_loss[-1]
    
    print("create_model() - Training loss:")
    print(training_loss)
    
    return model, scaler, training_loss

