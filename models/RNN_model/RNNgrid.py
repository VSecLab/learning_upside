import pandas as pd
import numpy as np
import VisorData as vd
import time as tm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

def model_and_evaluate(userid, sid, features):
    data = vd.get_all_users_()
    users = vd.get_users()
    
    username = users[userid]
    seqIDs = vd.get_SeqIDs_user(data, users[userid])
    dfs = vd.get_all_df_for_user(data, users[userid], features)
    df = dfs[seqIDs[sid]]

    def build_model():
        data = vd.get_all_users_()
        users = vd.get_users()
    
        username = users[userid]
        seqIDs = vd.get_SeqIDs_user(data, users[userid])
        dfs = vd.get_all_df_for_user(data, users[userid], features)
        df = dfs[seqIDs[sid]]
        df = df.dropna()

    # Normalizzazione dei dati
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)

    # Reshape per adattarsi all'input del modello RNN (necessario per la forma [samples, timesteps, features])
        df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))
        model = Sequential()
        model.add(SimpleRNN(units=32, input_shape=(df_scaled.shape[1], 1)))
        model.add(Dense(units=df.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model

    rnn_regressor = KerasRegressor(build_fn=build_model)

    param_grid = {
       'epochs': [2, 5, 10,15],
       'batch_size': [8, 16, 32]
    }
    
    df = df.dropna()

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Reshape per adattarsi all'input del modello RNN (necessario per la forma [samples, timesteps, features])
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))
    grid_search = GridSearchCV(estimator=rnn_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_result = grid_search.fit(df_scaled, df_scaled)

    print(f"Best score for user {username}, sequence ID {seqIDs[sid]}: {grid_result.best_score_} using {grid_result.best_params_}")

    return grid_result.best_score_, grid_result.best_params_

# Define lists to store the best scores and parameters
best_scores = []
best_params = []

# Loop over user IDs
for userid in range(0,4):  # Replace NUM_USERS with the actual number of users
    # Loop over sequence IDs for the current user
    for sid in range(20):  # Replace NUM_SEQS with the actual number of sequences
        best_score, best_param = model_and_evaluate(userid, sid, 'Head_Pitch')
        best_scores.append(best_score)
        best_params.append(best_param)

# Calculate the mean of the best scores
mean_best_score = np.mean(best_scores)

# Calculate the mean of the best parameters
mean_best_params = {}
for param in best_params[0].keys():
    param_values = [params[param] for params in best_params]
    mean_best_params[param] = np.mean(param_values)

print(f"Mean best score: {mean_best_score}")
print("Mean best parameters:")
print(mean_best_params)
