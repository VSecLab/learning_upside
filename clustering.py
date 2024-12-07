import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def clusterData():
    # Leggi i dati da un DataFrame
    percorso_file_csv = 'allUsers.csv'
    df = pd.read_csv(percorso_file_csv)

    # Seleziona solo le colonne numeriche per il clustering
    colonne_per_clustering = df.drop('Timestamp', axis=1)
    colonne_per_clustering = df.drop('user', axis=1)


    # Inizializza il modello K-Means
    numero_di_cluster = 5
    kmeans = KMeans(n_clusters=numero_di_cluster, random_state=42)

    # Esegui il clustering sulle colonne selezionate
    cluster_labels = kmeans.fit_predict(colonne_per_clustering)

    # Aggiungi le etichette di clustering al DataFrame
    df['Cluster'] = cluster_labels

    # Salva il DataFrame risultante in un nuovo file CSV
    #df.to_csv('dataset_dopo_clustering.csv', index=False)

    # Visualizza le prime righe del DataFrame con le etichette di clustering
    #print(df)
    return df

def analysis():
    # Visualizza il grafico PCA a due dimensioni con i colori basati sui cluster
    pca = PCA(n_components=2)
    riduzione_dimensioni = pca.fit_transform(colonne_per_clustering)
    df['Dim1'] = riduzione_dimensioni[:, 0]
    df['Dim2'] = riduzione_dimensioni[:, 1]

    # Crea un grafico a dispersione con colori basati sui cluster
    plt.scatter(df['Dim1'], df['Dim2'], c=df['Cluster'], cmap='viridis')
    plt.title('Cluster Visualization')
    plt.xlabel('Dimensione 1')
    plt.ylabel('Dimensione 2')
    plt.savefig('cluster.png')


    from sklearn.metrics import silhouette_score
    # Calcola il Silhouette Score
    silhouette_avg = silhouette_score(colonne_per_clustering, cluster_labels)
    # Stampa il risultato
    print(f'Punteggio Silhouette: {silhouette_avg}')
    cluster_centroids = kmeans.cluster_centers_
    print(f'Centroidi dei Cluster:\n{cluster_centroids}')

def countUsersInClusters(df):
    users=df['user'].unique()
    clusters=df['Cluster'].unique()
    #Iterate on clusters 
    for cluster in clusters:
        clusterdf=df.loc[df['Cluster'] == cluster]
        users=clusterdf['user'].unique()
        for user in users:
            select=df.loc[df['user'] == user]
            count=select.shape[0]
            print('Cluster:'+str(cluster)+',user:'+user+' count:'+str(count))



def UsersVSCLusters(df):
    # Crea un grafico a dispersione con colori basati sui cluster
    plt.scatter(df['user'], df['Cluster_Label'], c=df['Cluster'], cmap='viridis')
    plt.title('Cluster Visualization')
    plt.xlabel('user')
    plt.ylabel('Cluster_Label')
    plt.savefig('cluster.png')

    users=df['user'].unique()
    clusters=df['Cluster'].unique()
    print(users)
    print(clusters)

    
def _main_():
    df=clusterData()
    #UsersVSCLusters(df)
    countUsersInClusters(df)


_main_()