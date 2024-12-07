import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

matplotlib.use('agg')

def ml_kmeansPCA_plot(dataset, sensor, kfit):
    """
    Plots the results of KMeans clustering on a dataset reduced to 2D using PCA.
    
    :param numpy.ndarray dataset: the dataset to be plotted, expected to be a 2D array.
    :param str sensor: the name of the sensor or dataset, used for the plot title and filename.
    :param KMeans kfit: the fitted KMeans model containing the cluster centers.
    :return: the filename of the saved plot image.
    :rtype str
    """

    dataset = dataset.astype(float)
    plt.figure(figsize=(10, 5))
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=1)

    centers = kfit.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], alpha=1, c="red", marker="X")
    plt.title(sensor + " data point and cluster centers with PCA")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'PCA_{sensor}_kmeans_plot_{timestamp}.png'
    plt.savefig("static/images/" + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return plot_filename


def ml_kmeans_plot(dataset, sensor, kfit): 
    """
    Plots a 3D scatter plot of the dataset and KMeans cluster centers.
    
    :param numpy.ndarray dataset: the dataset to be plotted, expected to be a 2D array with 3 columns.
    :param str sensor: the name of the sensor, used for the plot title and filename.
    :param KMeans kfi: a fitted KMeans object containing the cluster centers.
    :return: the filename of the saved plot image.
    :rtype: str
    """
    dataset = dataset.astype(float)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection="3d")

    distances = np.linalg.norm(dataset, axis=1) # euclidean distance of each point from the origin (for coloring)

    centers = kfit.cluster_centers_
    ax.scatter3D(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        alpha=1,
        c="red",
        marker="X", 
        s = 100, 
        linewidth=2
    )

    ax.scatter3D(
        dataset[:, 0], dataset[:, 1], dataset[:, 2], c=distances, alpha=0.3, cmap="viridis"
    )
    plt.title(sensor + " 3D data point and cluster centers")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'{sensor}_kmeans_plot_{timestamp}.png'
    plt.savefig("static/images/" + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return plot_filename

def PCA_kmeans(df, sensor):
    """
    Perform PCA and KMeans clustering on the given dataframe and sensor data.
    This function first scales the data, then reduces its dimensionality using PCA,
    and finally applies KMeans clustering to the reduced data. It also generates a plot
    of the clustered data.
    :param pd.DataFrame df: the dataframe to be clustered. It should contain columns "varianceX", "varianceY", and "varianceZ".
    :param str sensor: the name of the sensor, used for labeling the plot.
    :return: two dataframes (one for each cluster) and the name of the plot file.
    :rtype: pd.DataFrame, pd.DataFrame, str
    """
    dataset = df[["varianceX", "varianceY", "varianceZ"]].to_numpy()
    dataset = StandardScaler().fit_transform(dataset)
    pca = PCA(n_components=2)

    reduced = pca.fit_transform(dataset)
    reduced_df = pd.DataFrame(data=reduced, columns=["component1", "component2"])
    red = reduced_df[["component1", "component2"]].to_numpy()

    kmeans = KMeans(n_clusters=2)
    kfit = kmeans.fit(red)
    res = kmeans.predict(red)

    plt_name = ml_kmeansPCA_plot(red, sensor, kfit)

    new_df = df
    new_df = new_df.reset_index()
    new_df.insert(6, "motion", res, allow_duplicates=True)

    new_df_1 = new_df[new_df["motion"] == 1]
    new_df_0 = new_df[new_df["motion"] == 0]

    return new_df_0, new_df_1, plt_name

def ml_kmeans(df, sensor):
    """
    Perform KMeans clustering on the given dataframe using the specified sensor data.
    This function clusters the data based on the variance of X, Y, and Z axes using KMeans with 2 clusters.
    It also generates a plot of the clustering result and returns two dataframes (one for each cluster) and the name of the plot file.
    
    :param pd.DataFrame df: the dataframe to be clustered.
    :param str sensor: the name of the sensor.
    :return: two dataframes (one for each cluster) and the name of the plot file.
    :rtype: pd.DataFrame, pd.DataFrame, str
    """
    dataset = df[["varianceX", "varianceY", "varianceZ"]].to_numpy()
    #print(dataset)
    kmeans = KMeans(n_clusters=2)
    kfit = kmeans.fit(dataset)
    res = kmeans.predict(dataset)

    plt_name = ml_kmeans_plot(dataset, sensor, kfit)
    
    #print(res)

    new_df = df
    new_df = new_df.reset_index()
    new_df.insert(6, "motion", res, allow_duplicates=True)

    new_df_1 = new_df[new_df["motion"] == 1]
    new_df_0 = new_df[new_df["motion"] == 0]

    #print(new_df_1)
    #print(new_df_0)

    return new_df_0, new_df_1, plt_name


if __name__ == "__main__": 
    pass