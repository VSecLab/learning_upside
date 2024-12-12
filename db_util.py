import pandas as pd
import mysql.connector
 #sql alchemy
def db_connect(): 
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="allDataDb"
    )
    mycursor = mydb.cursor()

    return mydb, mycursor

def db_disconnect(mydb, mycursor):
    mycursor.close()
    mydb.close()

def get_df_from_logID(logIDs):
    """
        Get the DataFrame for the given logIDs. 
        logIDs is a dictionary with the sensor as key and the list of logIDs as value.
        
        :param dict logIDs: dict of logIDs for the sensors
        :returns: DataFrame for the given logIDs and sensor
        :rtype: pd.DataFrame
    """
    mydb, mycursor = db_connect()

    logs_dict = {}
    #keys = list[logIDs.keys()]
    for sensor, id_logs in logIDs.items():
        for id_log in id_logs:
            sql = f"select X, Y, Z from {sensor.lower()} where ID_log = \"{id_log}\""
            try:
                mycursor.execute(sql)
                values = mycursor.fetchall()
                logs_dict[id_log] = pd.DataFrame(values, columns=["X", "Y", "Z"])
                mydb.commit()
            except mysql.connector.Error as err:
                print(f"Error - logs: {err}")
            print(f"get_df_from_logID - sensor {sensor} - log {id_log}: done")
    db_disconnect(mydb, mycursor)

    return logs_dict

def get_user_seqIDs(userID, activity): 
    """
        Get the sequence IDs for the given user and activity.
      
        :param int userID: ID of the User
        :param str activity: Name of the activity
        :returns: list of sequence IDs 
        :rtype: list
    """
    mydb, mycursor = db_connect()

    # grims: retrive the sequence IDs for the given user and activity
    sql = f"select distinct seqID from movimento as m where m.ID_utente = {userID} and ID_attivita = (select ID_attivita from attivita where nome_attivita = \"{activity}\")"
    
    try: 
        mycursor.execute(sql)
        seqIDs = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID_log: {err}")

    db_disconnect(mydb, mycursor)
    seqIDs = [seqID[0] for seqID in seqIDs]
    seqIDs.sort()
    #print(seqIDs)
    return seqIDs

def get_users(activity):
    """
        Retrive the names of all the users for the given activity.

        :param str activity: Name of the activity
        :returns: list of users for the given activity
        :rtype: list
    """
    mydb, mycursor = db_connect()

    sql = "SELECT u.Nome FROM utente u WHERE u.ID IN (SELECT DISTINCT e.ID_utente FROM esecuzione e JOIN attivita a ON e.ID_attivita = a.ID_attivita WHERE a.nome_attivita = (%s));"

    try: 
        mycursor.execute(sql, (activity,))
        result = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - get_users(): {err}")
    users = [row[0] for row in result]  # Assuming you want the first column of each row
    db_disconnect(mydb, mycursor)
    #print(users)
    return users

def get_user_log_onFeatures(userid, sid, device, features):
    """
        Get the DataFrame of the given user, for the given sequence number and the given features. 

        :param int userid: ID for the User
        :param str sid: sequence number
        :param list features: list of features to be used for the model 
        :returns: DataFrame for the given user and features
        :rtype: pd.DataFrame
        
    """
    orientation = ["Pitch", "Yaw", "Roll"]
    position = ["X", "Y", "Z"]
    features = features.split() # grims: if we want to implement the train over multiple features
    # grims: TODO aggiungere form per selezionare l'evento 
    event = "metalearning"
    mydb, mycursor = db_connect()

    # grims: create the model for the given user and sequence
    sql = "select UDI from dispositivo where tipo = (%s)"
    
    # grims: retrive the UDI of the device  
    try: 
        mycursor.execute(sql, (device,))
        udi = mycursor.fetchone()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - UDI: {err}") 

    # grims: retrive the sensor
    if any(element in orientation for element in features): 
        sensor = "Orientation_Sensor"
    elif any(element in position for element in features): 
        sensor = "Position_Sensor"
    
    print("SID: ", sid)
    # grims: retrive the ID_log
    #sql = f"select ID_movimento from movimento where ID_utente = {userid} and sensore = \"{sensor}\" and UDI = \"{udi[0]}\" and ID_movimento LIKE \"{str(sid)}\_%\""
    sql = f"select ID_movimento from movimento where ID_utente = {userid} and sensore = \"{sensor}\" and UDI = \"{udi[0]}\" and seqID = {sid}"
    
    try:    
        mycursor.execute(sql)
        id_log = mycursor.fetchone()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID_log: {err}")

    columns = ', '.join(features)  # Unisci i nomi delle colonne in una stringa separata da virgole
    #print(f"sensor: {sensor}; columns: {columns}")
    # grims: retrive the data and create the DataFrame for the model 
    if sensor == "Orientation_Sensor": 
        sql = f"select {columns} from orientation_sensor where ID_log = \"{id_log[0]}\""
    elif sensor == "Position_Sensor": 
        sql = f"select {columns} from position_sensor where ID_log = \"{id_log[0]}\""
    
    try: 
        mycursor.execute(sql)
        logs = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - logs: {err}")

    df = pd.DataFrame(logs, columns=[features])

    db_disconnect(mydb=mydb, mycursor=mycursor)

    return df

def get_userID(user): 
    """
        Get the ID for the given user.
        
        :param str user: name of the user
        :returns: ID for the given user 
        :rtype: int
    """

    mydb, mycursor = db_connect()

    sql = "select u.ID from utente as u where u.Nome = (%s)"
    
    try: 
        mycursor.execute(sql, (user,))
        id = mycursor.fetchone()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    return id[0]

def getAll_userdf_onFeatures(userID, seqIDs, device, features):
    """
        Get a dictionary of DataFrames for the given user, for the given sequence numbers and the given features.
        
        :param int user: userID of the user
        :param list seqIDs: list of sequence numbers
        :param str device: name of the device
        :param list features: list of features to be used for the model 
        :returns: dictionary of DataFrames for the given user and features
        :rtype: dict
    """

    userdf = {}
    print(features)

    for sid in seqIDs:
        userdf[sid] = get_user_log_onFeatures(userID, sid, device, features)

    #print(userdf)
    return userdf

if __name__ == "__main__":
    #get_user_seqIDs("AlessandroMercurio", "metalearning")
    #get_users("metalearning")
    """ids = get_user_seqIDs("AlessandroMercurio", "metalearning")
    getAll_userdf_onFeatures("82", ids, "visore", "Pitch")"""

    #get_userID("AlessandroMercurio")
    pass