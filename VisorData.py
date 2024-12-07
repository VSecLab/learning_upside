import pandas as pd
import os
import matplotlib.pyplot as plt

folder_path = "dati/"

def hello():
    print('hello')
    
def labelAdd(df, label, label_name):
    len=df.shape[0]
    label_list= [label]*len
    df[label_name]=label_list
    return df

def get_users():
    all_users =[]
    # Loop attraverso le cartelle degli utenti
    for username in os.listdir(folder_path):
        if username[0] == '.':
            print('skipping '+username)
        else:
            all_users.append(username)
    return all_users

def get_pins(df):
    pins=[]
    res=pd.unique(df['PINID'])
    return res

def get_pins_user(df,user):
    df=df.loc[df['user'] == user]
    res=get_pins(df)
    return res

def get_SeqIDs(df):
    res=pd.unique(df['SeqID'])
    return res

def get_SeqIDs_user(df,user):
    df=df.loc[df['user'] == user]
    seqIDs=get_SeqIDs(df)
    return seqIDs

def get_sequence(seqID,data):
    df1=data.loc[data['SeqID'] == seqID]
    return df1

def get_ts(data,seqID,parameter):
    df1=get_sequence(seqID,data)
    ts=df1[['Timestamp',parameter]].to_numpy()
    return ts

def get_all_ts_for_user(data,user):
    df=data.loc[data['user'] == user]
    pins=get_pins(df)
    seqIDs=get_SeqIDs(df)

    tss={}
    for id in seqIDs:
        tss[id]=get_ts(data,id,'Head_Pitch')
    return tss


def get_df(data,seqID,parameter):
    df1=get_sequence(seqID,data)
    ts=df1[['Timestamp',parameter]]
    return ts



def aligntime(df):
    start_time=df['Timestamp'].iloc[0]
    # print (start_time)
    df.loc[:, "Timestamp"] = df["Timestamp"].apply(lambda x: x - start_time)
    return df

def get_user_data(username):
    # Inizializza una lista per memorizzare tutti i DataFrame
    all_dataframes = []

    # Loop attraverso i file nella cartella
    user_path = os.path.join(folder_path, username)
    for filename in os.listdir(user_path):
        if filename.endswith(".csv"):
            # Costruisci il percorso completo del file
            file_path = os.path.join(user_path, filename)
            #print(filename[16:]+' '+str(filename[16:].find('_'))+' '+filename[16:16+filename[16:].find('_')])
            pin=filename[16:16+filename[16:].find('_')]
            seqID=filename[16:]
            # print(seqID)
            # Leggi il file CSV e aggiungi il DataFrame alla lista
            df = pd.read_csv(file_path)
            df2=labelAdd(df,pin,'PINID')
            df3=labelAdd(df,seqID,'SeqID')
            df4=aligntime(df3)
            all_dataframes.append(df3)
    
    # Concatena tutti i DataFrame in un unico DataFrame
    all_data = pd.concat(all_dataframes, ignore_index=True)
    userdf_labeled=labelAdd(all_data,username,'user')
    return userdf_labeled

def save_as_file(df,filename):
    # Salva il DataFrame consolidato come un nuovo file CSV
    df.to_csv(filename, index=False)

def get_all_users_():
    print('Getting users')
    users=get_users()
    all_users=[]
    for user in users:
        print('Get user'+user)
        tmp_userdf=get_user_data(user)
        all_users.append(tmp_userdf)
    all_users_df=pd.concat(all_users, ignore_index=True)
    return all_users_df


def _main_():
    usersdf=get_all_users_()
    usersdf.drop('Persona',axis='columns',inplace=True)
    print(usersdf.head())
    save_as_file(usersdf,'allUsers.csv')

def _test_():
    users=get_users()
    for user in users:
        userdf=get_user_data(user)
        plot(userdf,user)

def plot(df,filename):
    pins=df['PINID'].unique()
    # print(pins)

    #Plot Head Rotation
    ax1=plt.subplot(311)
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_Pitch')
        plt.scatter(tmpdf['Timestamp'], tmpdf['Head_Pitch'])
    ax2=plt.subplot(312,sharex=ax1)
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_Yaw')
        plt.scatter(tmpdf['Timestamp'], tmpdf['Head_Yaw'])
    ax3=plt.subplot(313,sharex=ax1)    
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_Roll')
        plt.scatter(tmpdf['Timestamp'], tmpdf['Head_Roll'])
    #plt.plot(tmpdf['Timestamp'], tmpdf['Head_Y'])
    #plt.plot(tmpdf['Timestamp'], tmpdf['Head_Z'])
    
    plt.xlabel('Timestamp')
    plt.savefig('plots/'+filename+'HR.png')
    plt.clf()
    #Plot Head Position
    ax1=plt.subplot(311)
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_X')
        plt.plot(tmpdf['Timestamp'], tmpdf['Head_X'])
    ax2=plt.subplot(312,sharex=ax1)
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_Y')
        plt.plot(tmpdf['Timestamp'], tmpdf['Head_Y'])
    ax3=plt.subplot(313,sharex=ax1)    
    for pin in pins:            
        tmpdf=df.loc[df['PINID'] == pin]
        plt.ylabel('Head_Z')
        plt.plot(tmpdf['Timestamp'], tmpdf['Head_Z'])
    #plt.plot(tmpdf['Timestamp'], tmpdf['Head_Y'])
    #plt.plot(tmpdf['Timestamp'], tmpdf['Head_Z'])
    
    plt.xlabel('Timestamp')
    plt.savefig('plots/'+filename+'HP.png')


def get_df_with_timestamp(data, seqID, parameters):
    df1 = get_sequence(seqID, data)
    if isinstance(parameters, str):  # Se parameters è una stringa, consideralo come un singolo parametro
        ts = df1[['Timestamp', parameters]]
    else:  # Altrimenti, parameters è una lista di parametri
        ts = df1[['Timestamp'] + parameters]
    print(ts)
    return ts

def get_all_df_for_user_with_timestamp(data, user, parameters):
    df = data.loc[data['user'] == user]
    pins = get_pins(df)
    seqIDs = get_SeqIDs(df)
    
    tdfs = {}
    for id in seqIDs:
        tdfs[id] = get_df_with_timestamp(data, id, parameters)
    return tdfs





def combine_user_csv(userid):
    userdf = get_user_data(userid)
    
    # Add sequence column
    seqIDs = get_SeqIDs_user(userdf, userid)
    userdf['SeqID'] = userdf.apply(lambda row: row['SeqID'] if row['SeqID'] in seqIDs else None, axis=1).ffill()

    # Save the combined DataFrame as a new CSV file
    combined_filename = f'{userid}_combined.csv'
    save_as_file(userdf, combined_filename)
    print(f"Combined CSV for {userid} saved as {combined_filename}")


def combine_first_10_csv(userid):
    userdf = pd.DataFrame()  # Initialize an empty DataFrame to store combined data
    user_path = os.path.join(folder_path, userid)
    
    # Loop through the first 10 CSV files in the user's folder
    for filename in sorted(os.listdir(user_path))[:10]:
        if filename.endswith(".csv"):
            file_path = os.path.join(user_path, filename)
            df = pd.read_csv(file_path)
            
            # Add SeqID column
            seqID = filename[16:]
            df = labelAdd(df, seqID, 'SeqID')
            
            # Align timestamps
            df = aligntime(df)
            
            # Concatenate to the overall DataFrame
            userdf = pd.concat([userdf, df], ignore_index=True)

    # Save the combined DataFrame as a new CSV file
    combined_filename = f'{userid}_combined_first_10.csv'
    save_as_file(userdf, combined_filename)
    print(f"Combined first 10 CSVs for {userid} saved as {combined_filename}")

# Example usage:
#combine_first_10_csv('ConteTeresa')





#combinare tutti i csv di ConteTeresa
#combine_user_csv('ConteTeresa')


#_main_()
#_test_()
    