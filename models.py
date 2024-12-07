from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, RepeatVector, TimeDistributed

def create_model(df, epochs, batch_size, model_type):
    # Remove rows with missing values
    df = df.dropna()

    # Check if the DataFrame is empty after dropping NA
    if df.empty:
        raise ValueError("The input DataFrame is empty after dropping missing values.")

    # Reshape to fit the model input (necessary for the shape [samples, timesteps, features])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Reshape to fit the model input (necessary for the shape [samples, timesteps, features])
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    # Model architecture configuration
    model = Sequential()

    if model_type == 'RNN' or model_type == 'RNN3label':
        model.add(SimpleRNN(units=32, input_shape=(df_scaled.shape[1], 1)))
        model.add(Dense(units=df_scaled.shape[1]))  # Output per ogni passo temporale
        print("------RNN model------")
    elif model_type == 'LSTMauto' or model_type == 'LSTMlabel':
        model.add(LSTM(units=32, input_shape=(df_scaled.shape[1], 1)))
        model.add(RepeatVector(df_scaled.shape[1]))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(TimeDistributed(Dense(units=1)))
        print("------LSTMauto model------")
    else:
        print(f"Model type '{model_type}' is not supported.")
        raise ValueError("Invalid model_type. Supported types are 'RNN', 'RNN3label', 'LSTMauto', 'LSTMlabel'.")

    model.compile(optimizer='adam', loss='mse')

    # Model training
    model.fit(df_scaled, df_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, scaler

