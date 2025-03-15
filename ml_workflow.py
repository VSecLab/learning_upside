import os
import random
import matplotlib
import time as tm
import numpy as np  
import pandas as pd 
import db_util as db
import seaborn as sns
import VisorData as vd
import matplotlib.pyplot as plt


from datetime import datetime
from models import create_model
from sklearn.metrics import mean_squared_error

matplotlib.use('agg')

def get_tp_fp_fn_tn(df, event_name):
    """
    Calculate the number of true positives (TP), false positives (FP), 
    false negatives (FN), and true negatives (TN) based on the given DataFrame and event name.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be evaluated.
    event_name : str
        The name of the event to be used for evaluation. 
        It can be either "UPSIDE" or "metalearning".

    Returns
    -------
    tuple
        A tuple containing four integers (TP, FP, FN, TN), representing the counts of 
        true positives, false positives, false negatives, and true negatives, respectively.

    Raises
    ------
    ValueError
        If the input DataFrame is empty.

    Notes
    -----
    This function evaluates the provided data to compute confusion matrix metrics 
    (TP, FP, FN, TN) based on the specified `event_name`. The event name determines 
    the criteria for evaluation.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    TP, FP, FN, TN = 0, 0, 0, 0
    for index, row in df.iterrows():
        if row['Recognised'] == 'Positive':
            if event_name == "UPSIDE":
                if row['activity'] == 'Positive':
                    TP += 1
                else:
                    FP += 1
            elif event_name == "metalearning":
                if row['auth_user'] == 'Positive':
                    TP += 1
                else:
                    FP += 1
        elif row['Recognised'] == 'Negative':
            if event_name == "UPSIDE":
                if row['activity'] == 'Positive':
                    FN += 1
                else:
                    TN += 1
            elif event_name == "metalearning":
                if row['auth_user'] == 'Positive':
                    FN += 1
                else:
                    TN += 1

    return TP, FP, FN, TN

def confusion_matrix(df, model_name, sensor, filename, event_name):
    """
    Generates and saves a confusion matrix for a given model and dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    model_name : str
        The name of the model.
    sensor : str
        The sensor used for data collection.
    filename : str
        The name of the file to save the confusion matrix.
    event_name : str
        The name of the event being analyzed.

    Returns
    -------
    str
        The relative path to the saved confusion matrix image.

    Raises
    ------
    ValueError
        If the input DataFrame is empty.

    Notes
    -----
    The function calculates the true positives (TP), false positives (FP), 
    false negatives (FN), and true negatives (TN) from the DataFrame using 
    the `get_tp_fp_fn_tn` function. It then constructs a confusion matrix, 
    saves it as an image file, and returns the relative path to the saved image.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    tp, fp, fn, tn = get_tp_fp_fn_tn(df, event_name)
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')

    cm = np.array([[tp, fn], [fp, tn]])  # Costruisci la matrice di confusione direttamente
    print('Confusion Matrix:')
    print(cm)
    directory = f'static/images/{event_name}_grims_confusion_matrix_{model_name}'
    os.makedirs(directory, exist_ok=True)
    path = f'{directory}/{event_name}_{sensor}_{filename}_confusion_matrix.png'

    print(f'Confusion Matrix saved at: {path}')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap = 'crest', fmt='d', xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path)
    plt.close()

    partial_path = f'{event_name}_grims_confusion_matrix_{model_name}/{event_name}_{sensor}_{filename}_confusion_matrix.png'
    return partial_path

def evalutate_model_on_activity(right_test_logs, wrong_test_logs, chosen_model, model_name, scaler):
    """
    Evaluates a given model on right and wrong activity logs and saves the results to CSV files.

    Parameters
    ----------
    right_test_logs : list
        List of log IDs for the right activity.
    wrong_test_logs : list
        List of log IDs for the wrong activity.
    chosen_model : object
        The machine learning model to be evaluated.
    model_name : str
        The name of the model, used for naming the result files.
    scaler : object
        The scaler used to preprocess the data before evaluation.

    Returns
    -------
    str
        The path to the CSV file containing the evaluation results.
    str
        The base name of the file.
    """
    right_logs = db.get_df_from_logID(right_test_logs)

    wrong_logs = db.get_df_from_logID(wrong_test_logs)

    right_schema = {'log_id': [], 'mse': [], 'activity': 'Positive'}
    right_mses = pd.DataFrame(right_schema)

    wrong_schema = {'log_id': [], 'mse': [], 'activity': 'Negative'}
    wrong_mses = pd.DataFrame(wrong_schema)

    num_entry = 100

    print('Right logs evaluation')
    for log in right_logs:
        print(f'Log evaluation: {log}')
        df = right_logs[log]
        if len(df) < num_entry:
            print(f'Log {log} has less than {num_entry} entries, skipping.')
            continue
        mse = eval(chosen_model, model_name, scaler, df, log)
        print(f'Mean Squared Error: {mse}')
        right_mses.loc[len(right_mses)] = [log, mse, 'Positive']
        
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f'ml_result/grims_resultsMse_{model_name}'
    os.makedirs(directory, exist_ok=True)
    file=f'{directory}/{model_name}_mse_' + timestamp + '.csv'
    right_mses.to_csv(file, index=False)
    
    print('\nWrong logs evaluation')
    for log in wrong_logs:
        print(f'Log evaluation: {log}')
        df = wrong_logs[log]
        if len(df) < num_entry: # 100 is the timestep
            print(f'Log {log} has less than {num_entry} entries, skipping.')
            continue
        mse = eval(chosen_model, model_name, scaler, df, log)
        print(f'Mean Squared Error: {mse}')
        wrong_mses.loc[len(wrong_mses)] = [log, mse, 'Negative']
    
    directory = f'ml_result/grims_resultsMse_{model_name}'
    os.makedirs(directory, exist_ok=True)
    file=f'{directory}/{model_name}_mse_' + timestamp + '.csv'
    if os.path.exists(file):
        wrong_mses.to_csv(file, mode='a', header=False, index=False)
    else:
        wrong_mses.to_csv(file, index=False)
    basename = f'{model_name}_mse_' + timestamp
    
    return file, basename

def eval_models(sensor, threshold, right_test_logs, wrong_test_logs, chosen_model, model_name, scaler, event_name):
    """
    Evaluate the performance of a machine learning model on a given activity and compute the confusion matrix.

    Parameters
    ----------
    sensor : str
        The sensor type used for evaluation.
    threshold : float
        The threshold value for determining recognition.
    right_test_logs : str
        Path to the logs of correctly classified activities.
    wrong_test_logs : str
        Path to the logs of incorrectly classified activities.
    chosen_model : object
        The machine learning model to be evaluated.
    model_name : str
        The name of the machine learning model.
    scaler : object
        The scaler used for data normalization.
    event_name : str
        The name of the event being evaluated.

    Returns
    -------
    dict
        A dictionary containing the mean squared error (MSE) for each log.
    str
        The path to the confusion matrix.
    """

    filename, basename = evalutate_model_on_activity(right_test_logs, wrong_test_logs, chosen_model, model_name, scaler)
    #print("\nvalidate_logs_page() - filename", filename)

    # Open the filename as a CSV file with pandas
    df = pd.read_csv(filename)
    df['Recognised'] = df['mse'].apply(lambda x: 'Positive' if x < threshold else 'Negative')
    df.to_csv(filename, index=False)

    # Save the results in a dictionary
    eval_results = {}
    for index, row in df.iterrows():
        eval_results[row['log_id']] = {
            'mse': row['mse'],
            'activity': row['activity'],
            'Recognised': row['Recognised']
        }
    
    conf_matrix = confusion_matrix(df, model_name, sensor, basename, event_name)
    return eval_results, conf_matrix

def model_train_lab_activity(lab_logs, activity_name, epochs, batch_size, percentage, model): 
    """
    Train a machine learning model on laboratory data.

    Parameters
    ----------
    logs : dict
        Dictionary with the sensor as key and the list of logIDs as value.
    activity_name : str
        The activity name to be used for training the model.
    epochs : int
        Number of epochs to train the model.
    batch_size : int
        Size of the batches used in training.
    model : str
        The type of model to be created and trained.

    Returns
    -------
    keras.Model
        The trained machine learning model.
    sklearn.preprocessing.StandardScaler
        The scaler used to normalize the features.
    float
        The training loss of the model.
    list
        List of log identifiers used for testing.
    list
        List of log identifiers used for training.
    """

    chosen_model, scaler, training_loss, test_log_keys, train_log_keys = model_train_activity(lab_logs, epochs, batch_size, percentage, model)

    return chosen_model, scaler, training_loss, test_log_keys, train_log_keys

def model_train_activity(logs, epochs, batch_size, percentage, model):
    """
    Trains a machine learning model using a specified percentage of log data for training.

    Parameters
    ----------
    logs : dict
        Dictionary with the sensor as key and the list of logIDs as value.
    epochs : int
        Number of epochs to train the model.
    batch_size : int
        Size of the batches used in training.
    percentage : int
        Percentage of the log data to be used for training.
    model : str
        The type of model to be created and trained.

    Returns
    -------
    keras.Model
        The trained machine learning model.
    sklearn.preprocessing.StandardScaler
        The scaler used to normalize the features.
    float
        The training loss of the model.
    list
        List of log identifiers used for testing.
    list
        List of log identifiers used for training.
    """

    logs_dict = db.get_df_from_logID(logs)

    #print(logs_dict)

    num_dataframes = len(logs_dict)
    
    x = (num_dataframes * percentage) // 100
    if x == 0:
        x = 1

    # randomize the order of the dictionary
    keys = list(logs_dict.keys())
    random.shuffle(keys)
    randomized_logs_dict = {key: logs_dict[key] for key in keys}
    #print(randomized_logs_dict)
    
    train_logs = {k: randomized_logs_dict[k] for k in list(randomized_logs_dict)[:x]}
    test_logs = {k: randomized_logs_dict[k] for k in list(randomized_logs_dict)[x:]}
    train_log_keys = list(train_logs.keys())
    test_log_keys = list(test_logs.keys())
    
    print(f'Train logs: {len(train_logs)}')
    print(f'Test logs: {len(test_logs)}')

    
    df = pd.concat(train_logs.values(), ignore_index=True)

    chosen_model, scaler, training_loss = create_model(df, epochs, batch_size, model)

    return chosen_model, scaler, training_loss, test_log_keys, train_log_keys

def model_train(userid, sid, device, features, epochs, batch_size, model):
    """
    Trains a machine learning model based on user-specific data and parameters.

    Parameters
    ----------
    userid : int
        The ID of the user for whom the model is being trained.
    sid : str
        The session ID associated with the user.
    device : str
        The device from which the data is collected.
    features : list
        A list of features to be used for training the model.
    epochs : int
        The number of epochs for training the model.
    batch_size : int
        The batch size for training the model.
    model : str
        The type of model to be trained.

    Returns
    -------
    keras.Model
        The trained machine learning model.
    sklearn.preprocessing.StandardScaler
        The scaler used to normalize the features.
    float
        The training loss of the model.
    """


    df = db.get_user_log_onFeatures(userid, sid, device, features)
    chosen_model, scaler, training_loss = create_model(df, epochs, batch_size, model)
    return chosen_model, scaler, training_loss

def evaluate_model(chosen_model, scaler, device, features, eval_sid, event_name, model_name):
    """
    Evaluate the chosen model on the given features and evaluation sequence IDs for each user.

    Parameters
    ----------
    chosen_model : object
        The machine learning model to be evaluated.
    scaler : object
        The scaler used to normalize the features.
    device : str
        The device identifier used for evaluation.
    features : list
        The list of features to be used for evaluation.
    eval_sid : list
        The list of evaluation sequence IDs.
    event_name : str
        The name of the event being evaluated.
    model_name : str
        The name of the model being evaluated.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the mean squared error (MSE) for each user and sequence.
    str 
        The file path where the results are saved.
    str
        The base name of the file.
    """
    
    users = db.get_users(event_name) # grims: TODO implement the activity selection

    # Valutare l'errore quadratico medio (MSE) per tutte le sequenze
    schema = {'sid': [], 'user': [], 'mse': []}
    mses = pd.DataFrame(schema)
    
    for user in users:
        print('User evaluation: ' + user)
        userID = db.get_userID(user) 
        #seqIDs = db.get_user_seqIDs(userID, "metalearning") # grims: TODO implement the activity selection 
        dfs = db.getAll_userdf_onFeatures(userID, eval_sid, device, features)
        
        # grims: TODO continua qua 
        for seqid in eval_sid:
            df = dfs[seqid]
            print(f'Valutazione utente: {user} sulla sequenza: {seqid}')
            #tm.sleep(1)
            mse = eval(chosen_model, scaler, df)
            print(f'User: {user}, seqID: {seqid}, Mean Squared Error: {mse}')
            mses.loc[len(mses)] = [seqid, user, mse]
        print('Next user')

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f'ml_result/metalearning_grims_resultsMse_{model_name}'
    os.makedirs(directory, exist_ok=True)
    file=f'{directory}/{model_name}_mse_' + timestamp + '.csv'
    if os.path.exists(file):
        mses.to_csv(file, mode='a', header=False, index=False)
    else:
        mses.to_csv(file, index=False)
    basename = f'{model_name}_mse_' + timestamp

    return mses, file, basename 

def meta_eval_models(chosen_model, threshold, userid, scaler, device, features, eval_sid, event_name, model_name): 
    """
    Evaluate a machine learning model and generate evaluation metrics and a confusion matrix.

    Parameters
    ----------
    chosen_model : object
        The machine learning model to be evaluated.
    threshold : float
        The threshold value for recognizing events.
    userid : int
        The user ID for fetching the username.
    scaler : object
        The scaler used for normalizing the features.
    device : str
        The device used for evaluation.
    features : list
        The list of features used in the model.
    eval_sid : int
        The session ID for evaluation.
    event_name : str
        The name of the event being evaluated.
    model_name : str
        The name of the model being evaluated.

    Returns
    -------
    dict
        A dictionary containing the mean squared error (MSE) for each user and sequence.
    dict
        A dictionary containing the evaluation results.
    str
        The path to the confusion matrix image.
        
    Notes
    -----
    The function evaluates the specified machine learning model using the given parameters. 
    It calculates metrics such as mean squared errors (MSE) and generates a confusion matrix. 
    The results are saved to a file, and the file path is returned.
    """
    mses, filename, basename = evaluate_model(chosen_model, scaler, device, features, eval_sid, event_name, model_name)

    df = pd.read_csv(filename)
    df['Recognised'] = df['mse'].apply(lambda x: 'Yes' if x < threshold else 'No')
    username = db.get_username(userid)
    df['auth_user'] = df['user'].apply(lambda x: 'right' if x == username else 'wrong')
    df.to_csv(filename, index=False)

    if features in ['Pitch', 'Roll', 'Yaw']:
        sensor = 'Orientation_Sensor'
    else: 
        sensor = 'Position_Sensor'

    path = confusion_matrix(df, model_name, sensor, basename, event_name)
     

    df_dict = df.groupby(df.index).apply(lambda x: x.to_dict(orient='records')).to_dict()

    return mses, df_dict, path

def eval(model, model_name, scaler, df, log): 
    """
    Evaluate the model on the given dataframe.

    Parameters
    ----------
    model : keras.Model
        The machine learning model to be evaluated.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to normalize the features.
    df : pandas.DataFrame
        The dataframe to evaluate the model on.

    Returns
    -------
    float
        The mean squared error (MSE) for the given dataframe.
    """
    timesteps = 100
    df = df.dropna()

    # data normalization
    df_scaled = scaler.transform(df)
    
    if model_name == 'LSTMlabel':
        x_test, y_train = create_sequence(df_scaled, timesteps)
        predictions_scaled = model.predict(x_test)

        # Save y_train and predictions_scaled to a CSV file
        results_df = pd.DataFrame({
            'y_train': y_train.flatten(),
            'predictions_scaled': predictions_scaled.flatten()
        })
        os.makedirs('ml_result/predictions_good_log', exist_ok=True)
        results_df.to_csv(f'ml_result/predictions_good_log/{log}_y_train_predictions_scaled_LSTM.csv', index=False)

        mse = mean_squared_error(y_train, predictions_scaled)

    elif model_name == 'LSTMauto':
        x_test = create_sequence_autoencoder(df_scaled, timesteps)
        predictions_scaled = model.predict(x_test)

        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        predictions_flattened = predictions_scaled.reshape(predictions_scaled.shape[0], -1)

        # Save x_test and predictions_scaled to a CSV file
        results_df = pd.DataFrame({
            'x_test': x_test_flattened.flatten(),
            'predictions_scaled': predictions_flattened.flatten()
        })
        os.makedirs('ml_result/predictions', exist_ok=True)
        results_df.to_csv(f'ml_result/predictions/{log}_x_test_predictions_scaled_LSTMauto.csv', index=False)

        mse = mean_squared_error(x_test_flattened, predictions_flattened)
    
    return mse
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