import random
import time as tm
import numpy as np  
import pandas as pd 
import db_util as db
import VisorData as vd

from models import create_model

#TODO this function 

def evalutate_model_on_activity(right_test_logs, wrong_test_logs, chosen_model, scaler):
    pass


def model_train_activity(logs, epochs, batch_size, model):
    logs_dict = db.get_df_from_logID(logs)

    #print(logs_dict)

    num_dataframes = len(logs_dict)

    x = (num_dataframes * 80) // 100
    
    # randomize the order of the dictionary
    keys = list(logs_dict.keys())
    random.shuffle(keys)
    randomized_logs_dict = {key: logs_dict[key] for key in keys}
    #print(randomized_logs_dict)
    
    train_logs = {k: randomized_logs_dict[k] for k in list(randomized_logs_dict)[:x]}
    test_logs = {k: randomized_logs_dict[k] for k in list(randomized_logs_dict)[x:]}
    test_log_keys = list(test_logs.keys())

    df = pd.concat(train_logs.values(), ignore_index=True)

    chosen_model, scaler = create_model(df, epochs, batch_size, model)

    return chosen_model, scaler, test_log_keys

def model_train(userid, sid, device, features, epochs, batch_size, model):
    """
    Trains a machine learning model based on user-specific data and parameters.

    :param int userid: the ID of the user for whom the model is being trained.
    :param str sid: the session ID associated with the user.
    :param str device: the device from which the data is collected.
    :param list features: a list of features to be used for training the model.
    :param int epochs: the number of epochs for training the model.
    :param int batch_size: the batch size for training the model.
    :param str model: the type of model to be trained.

    :return: a tuple containing the trained model and the scaler used for feature scaling.
    :rtype: tuple
    """


    df = db.get_user_log_onFeatures(userid, sid, device, features)
    chosen_model, scaler = create_model(df, epochs, batch_size, model)
    return chosen_model, scaler

    """directory = f'resultsMse_{model}_grims_db'
    os.makedirs(directory, exist_ok=True)
    file=f'{directory}/{model}_{features}_{epochs}_{batch_size}_mse' + userid + '_' + str(sid) + '.csv'
    mses.to_csv(file, index=False)"""

def evaluate_model(chosen_model, scaler, device, features, eval_sid):
    """
    Evaluate the chosen model on the given features and evaluation sequence IDs for each user.
    
    :param object chosen_model: the machine learning model to be evaluated.
    :param object scaler: the scaler used to normalize the features.
    :param str device: the device identifier used for evaluation.
    :param list features: the list of features to be used for evaluation.
    :param list eval_sid: the list of evaluation sequence IDs.
    :return: a DataFrame containing the mean squared error (MSE) for each user and sequence. 
    :rtype: pd.DataFrame
    """
    
    users = db.get_users("metalearning") # grims: TODO implement the activity selection

    # Valutare l'errore quadratico medio (MSE) per tutte le sequenze
    schema = {'user': [], 'mse': []}
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
            mses.loc[len(mses)] = [user, mse]
        print('Next user')

    return mses

def eval(model, scaler, df): 
    """
        Evaluate the model on the given dataframe.

        :param keras.Model model: the machine learning model to be evaluated.
        :param sklearn.preprocessing.StandardScaler scaler: the scaler used to normalize the features.
        :param pd.DataFrame df: the dataframe to evaluate the model on.
        :return: the mean squared error (MSE) for the given dataframe.
        :rtype: float
    """
    df = df.dropna()

    # data normalization
    df_scaled = scaler.transform(df)

    # reshape the data for the model input 
    df_scaled = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

    predictions_scaled = model.predict(df_scaled)

    # compute the mean squared error 
    mse = np.mean(np.power(df_scaled - predictions_scaled[:, :, np.newaxis], 2))

    return mse

if __name__ == "__main__":
    model_train(82, 1, "visore", "Pitch", 3, 2, "RNN")