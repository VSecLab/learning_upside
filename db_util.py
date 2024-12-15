import pandas as pd
import mysql.connector
 #sql alchemy
def db_connect(): 
    """
    Establishes a connection to the MySQL database and returns the connection and cursor objects.

    Returns
    -------
    mydb : mysql.connector.connection.MySQLConnection
        The MySQL database connection object.
    mycursor : mysql.connector.cursor.MySQLCursor
        The MySQL cursor object for executing queries.
    """
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="allDataDb"
    )
    mycursor = mydb.cursor()

    return mydb, mycursor

def db_disconnect(mydb, mycursor):
    """
    Closes the database cursor and connection.

    Parameters
    ----------
    mydb : mysql.connector.connection.MySQLConnection
        The database connection object to be closed.
    mycursor : mysql.connector.cursor.MySQLCursor
        The database cursor object to be closed.

    Returns
    -------
    None
    """
    mycursor.close()
    mydb.close()

def get_df_from_logID(logIDs):
    """
    Get the DataFrame for the given logIDs.

    Parameters
    ----------
    logIDs : dict
        Dictionary with the sensor as key and the list of logIDs as value.

    Returns
    -------
    dict
        Dictionary of DataFrames for the given logIDs and sensor.
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

    Parameters
    ----------
    userID : int
        ID of the User.
    activity : str
        Name of the activity.

    Returns
    -------
    list
        List of sequence IDs.
    """
    mydb, mycursor = db_connect()

    # grims: retrive the sequence IDs for the given user and activity
    sql = f"select distinct seqID from movement as m where m.ID_user = {userID} and ID_activity = (select ID_activity from activity where activity_name = \"{activity}\")"
    
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
    Retrieve the names of all the users for the given activity.

    Parameters
    ----------
    activity : str
        Name of the activity.

    Returns
    -------
    list
        List of users for the given activity.
    """
    mydb, mycursor = db_connect()

    sql = "SELECT u.Name FROM user u WHERE u.ID IN (SELECT DISTINCT e.ID_user FROM execution e JOIN activity a ON e.ID_activity = a.ID_activity WHERE a.activity_name = (%s));"

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

    Parameters
    ----------
    userid : int
        ID for the User.
    sid : str
        Sequence number.
    device : str
        UDI of the device.
    features : list
        List of features to be used for the model.

    Returns
    -------
    pd.DataFrame
        DataFrame for the given user and features.
    """
    orientation = ["Pitch", "Yaw", "Roll"]
    position = ["X", "Y", "Z"]
    features = features.split() # grims: if we want to implement the train over multiple features
    # grims: TODO aggiungere form per selezionare l'evento 
    event = "metalearning"
    mydb, mycursor = db_connect()

    # grims: create the model for the given user and sequence
    sql = "select UDI from device where type = (%s)"
    
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
    sql = f"select ID_movement from movement where ID_user = {userid} and sensor = \"{sensor}\" and UDI = \"{udi[0]}\" and seqID = {sid}"
    
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

    Parameters
    ----------
    user : str
        Name of the user.

    Returns
    -------
    int
        ID for the given user.
    """

    mydb, mycursor = db_connect()

    sql = "select u.ID from user as u where u.Name = (%s)"
    
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
    Get a dictionary of DataFrames for the given user, sequence numbers, and features.

    Parameters
    ----------
    userID : int
        User ID of the user.
    seqIDs : list
        List of sequence numbers.
    device : str
        Name of the device.
    features : list
        List of features to be used for the model.

    Returns
    -------
    dict
        Dictionary of DataFrames for the given user and features.
    """

    userdf = {}
    print(features)

    for sid in seqIDs:
        userdf[sid] = get_user_log_onFeatures(userID, sid, device, features)

    #print(userdf)
    return userdf

def get_eventID_by_event_name(event_name): 
    """
    Get the ID for the given event.

    Parameters
    ----------
    event_name : str
        Name of the event.

    Returns
    -------
    list
        List of event IDs for the given event name.
    """

    mydb, mycursor = db_connect()

    sql = f"SELECT ID_event FROM event where event_name = \"{event_name}\""
    
    try: 
        mycursor.execute(sql)
        id = mycursor.fetchone()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    return id[0]

def get_userIDs_by_eventID(event_id): 
    """
    Get the list of userIDs for the given eventID.

    Parameters
    ----------
    event_id : int
        ID of the event.

    Returns
    -------
    list
        List of userIDs for the given eventID.
    """

    mydb, mycursor = db_connect()

    try: 
        mycursor.execute(f"select ID_user from participation where ID_event = {event_id}")
        ids = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    ids = [row[0] for row in ids]

    return ids

def get_userIDs_and_movementID_by_eventID(event_id): 
    """
    Get the list of userIDs and movementIDs for the given eventID.

    Parameters
    ----------
    event_id : int
        ID of the event.

    Returns
    -------
    dict
        Dictionary with userIDs as keys and lists of movementIDs as values.
    """

    mydb, mycursor = db_connect()
    sql = f"select ID_user, ID_movement from movement where ID_user in (select ID_user from participation where ID_event = {event_id})"

    try: 
        mycursor.execute(sql)
        user_movement_list = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    # Convert list to dictionary
    user_movement = {}
    for user_id, movement_id in user_movement_list:
        if user_id not in user_movement:
            user_movement[user_id] = []
        user_movement[user_id].append(movement_id)

    return user_movement

def get_activityName_by_eventID(event_id):
    """
    Get the list of activities for the given event ID.

    Parameters
    ----------
    event_id : int
        ID of the event.

    Returns
    -------
    list
        List of activities corresponding to the given event ID.
    """

    mydb, mycursor = db_connect()
    
    try: 
        mycursor.execute(f"select activity_name from activity where ID_activity in (select a.ID_activity from association as a join event as e on a.ID_event and a.ID_event = {event_id})")
        activity_name = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    activity = [row[0] for row in activity_name] 

    return activity

def get_devices_by_eventID(event_id): 
    """
    Get the list of devices for the given event ID.

    Parameters
    ----------
    event_id : int
        ID of the event.

    Returns
    -------
    list
        List of devices for the given event ID.
    """

    mydb, mycursor = db_connect()
    
    try: 
        mycursor.execute(f"select distinct type from device as d where d.ID_activity in (select distinct a.ID_activity from association as a join event as e on a.ID_event and a.ID_event = {event_id})")
        devices = mycursor.fetchall()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    devices = [row[0] for row in devices]
    return devices

def get_username(userid): 
    """
    Get the name for the given user.

    Parameters
    ----------
    userid : int
        ID of the user.

    Returns
    -------
    str
        Name corresponding to the given user ID.
    """

    mydb, mycursor = db_connect()

    sql = f"select Name from user where ID = {userid}"
    
    try: 
        mycursor.execute(sql)
        name = mycursor.fetchone()
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error - ID: {err}")
    db_disconnect(mydb, mycursor)

    return name[0]

if __name__ == "__main__":
    #get_user_seqIDs("AlessandroMercurio", "metalearning")
    #get_users("metalearning")
    """ids = get_user_seqIDs("AlessandroMercurio", "metalearning")
    getAll_userdf_onFeatures("82", ids, "visore", "Pitch")"""

    #get_userID("AlessandroMercurio")
    pass