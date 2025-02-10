import os
import time as tm 
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

matplotlib.use('agg')
def plot_pca_iterative_kmeans(df1, df0, sensor, iteration):
    df1 = df1.astype(float)
    df0 = df0.astype(float)
    # Plotting in 3D
    fig = plt.figure()

    plt.scatter(df1[:, 0], df1[:, 1],  c='r', marker='o', label='Cluster 1')
    plt.scatter(df0[:, 0], df0[:, 1],  c='b', marker='^', label='Cluster 0')

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'PCA_{sensor}_{iteration}_kmeans_plot_{timestamp}.png'
    plot_dir = "static/images/kmeans_plt/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return plot_filename

def plot_iterative_kmeans(df1, df0, sensor, iteration):
    df1 = df1.astype(float)
    df0 = df0.astype(float)
    # Plotting in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming the DataFrames have columns 'x', 'y', 'z' for 3D plotting
    ax.scatter(df1[:, 0], df1[:, 1], df1[:, 2], c='r', marker='o', label='Cluster 1')
    ax.scatter(df0[:, 0], df0[:, 1], df0[:, 2], c='b', marker='^', label='Cluster 0')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'{sensor}_{iteration}_kmeans_plot_{timestamp}.png'
    plot_dir = "static/images/kmeans_plt/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return plot_filename

def ml_kmeansPCA_plot(dataset, sensor, kfit, model, iteration):
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
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=1, c=model.labels_.astype(float))

    centers = kfit.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], alpha=1, c="red", marker="X")
    plt.title(sensor + " data point and cluster centers with PCA")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'PCA_{sensor}_{iteration}_kmeans_plot_{timestamp}.png'
    plot_dir = "static/images/plots/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig("static/images/plots/" + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return


def ml_kmeans_plot(dataset, sensor, kfit, iteration): 
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

    # Assign colors based on cluster labels
    labels = kfit.labels_
    colors = ['blue' if label == 0 else 'green' for label in labels]

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
        dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colors, alpha=0.3
    )
    plt.title(sensor + " 3D data point and cluster centers")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f'{sensor}_{iteration}_kmeans_plot_{timestamp}.png'
    plot_dir = "static/images/plots/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.savefig("static/images/plots/" + plot_filename, pad_inches= 1, bbox_inches='tight')
    plt.close()

    return plot_filename

def PCA_kmeans(df, sensor, num_iterations):
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

    all_new_df_1 = pd.DataFrame()
    n_new_df_1 = pd.DataFrame()
    for i in range(num_iterations):
        print(i)
        dataset = df[["varianceX", "varianceY", "varianceZ"]].to_numpy()
        dataset = StandardScaler().fit_transform(dataset)
        pca = PCA(n_components=2)

        reduced = pca.fit_transform(dataset)
        reduced_df = pd.DataFrame(data=reduced, columns=["component1", "component2"])
        red = reduced_df[["component1", "component2"]].to_numpy()

        kmeans = KMeans(n_clusters=2, random_state=100)
        kfit = kmeans.fit(red)
        res = kmeans.predict(red)

        # Ensure that the larger values are always in cluster 1
        cluster_0_mean = red[res == 0].mean()
        cluster_1_mean = red[res == 1].mean()
        if cluster_0_mean > cluster_1_mean:
            res = 1 - res  # Swap cluster labels


        plt_name = ml_kmeansPCA_plot(red, sensor, kfit, kmeans, i + 1)

        new_df = df
        plt_df = reduced_df
        if "motion" in new_df.columns:
            new_df["motion"] = res
            new_df["iteration"] = i + 1

            plt_df["motion"] = res
            plt_df["iteration"] = i + 1
        else:
            new_df = new_df.reset_index()
            new_df.insert(6, "motion", res, allow_duplicates=True)
            new_df.insert(7, "iteration", i, allow_duplicates=True)

            plt_df = plt_df.reset_index()
            plt_df.insert(3, "motion", res, allow_duplicates=True)
            plt_df.insert(4, "iteration", i, allow_duplicates=True)

        new_df_1 = new_df[new_df["motion"] == 1]
        plt_df_1 = plt_df[plt_df["motion"] == 1]
        
        # Concatenate the new motion dataframes
        all_new_df_1 = pd.concat([all_new_df_1, plt_df_1], ignore_index=True) 
        n_new_df_1 = pd.concat([n_new_df_1, new_df_1], ignore_index=True)
        
        new_df_0 = new_df[new_df["motion"] == 0]
        plt_df_0 = plt_df[plt_df["motion"] == 0]

        df = new_df_0

    df1 = all_new_df_1[["component1", "component2"]].to_numpy()
    df0 = plt_df_0[["component1", "component2"]].to_numpy()

    plot = plot_pca_iterative_kmeans(df1, df0, sensor, i + 1)

    return new_df_0, n_new_df_1, plot

def ml_kmeans(df, sensor, num_iterations):
    """
    Perform KMeans clustering on the given dataframe for a specified number of iterations.
    This function applies KMeans clustering on the dataframe based on the variance of X, Y, and Z columns.
    It iteratively clusters the data, updates the dataframe with the cluster labels, and separates the data
    into two clusters. It also generates and saves a plot for each iteration.    

    :param pd.DataFrame df: the dataframe to be clustered. It should contain columns 'varianceX', 'varianceY', and 'varianceZ'.
    :param str sensor: the name of the sensor.
    :param int num_iterations: the number of times to run KMeans clustering.    
    :return: two dataframes (one for each cluster) and the name of the plot file.
    :rtype: pd.DataFrame, pd.DataFrame, str
    """

    all_new_df_1 = pd.DataFrame()
    for i in range(num_iterations):
        print(i)
        dataset = df[["varianceX", "varianceY", "varianceZ"]].to_numpy()
        #print(dataset)
        kmeans = KMeans(n_clusters=2, random_state=100)
        kfit = kmeans.fit(dataset)
        res = kmeans.predict(dataset)

         # Ensure that the larger values are always in cluster 1
        cluster_0_mean = dataset[res == 0].mean()
        cluster_1_mean = dataset[res == 1].mean()
        if cluster_0_mean > cluster_1_mean:
            res = 1 - res  # Swap cluster labels


        plt_name = ml_kmeans_plot(dataset, sensor, kfit, i + 1)
        
        #print(res)

        new_df = df
        
        if "motion" in new_df.columns:
            new_df["motion"] = res
            new_df["iteration"] = i + 1
        else:
            new_df = new_df.reset_index()
            new_df.insert(6, "motion", res, allow_duplicates=True)
            new_df.insert(7, "iteration", i, allow_duplicates=True)
        
        #print(new_df)

        new_df_1 = new_df[new_df["motion"] == 1]
        # Concatenate the new motion dataframes
        all_new_df_1 = pd.concat([all_new_df_1, new_df_1], ignore_index=True) 

        new_df_0 = new_df[new_df["motion"] == 0]

        df = new_df_0

    df1 = all_new_df_1[["varianceX", "varianceY", "varianceZ"]].to_numpy()
    df0 = new_df_0[["varianceX", "varianceY", "varianceZ"]].to_numpy()

    plot = plot_iterative_kmeans(df1, df0, sensor, i + 1)

    #print(new_df_1)
    #print(new_df_0)

    return new_df_0, all_new_df_1, plot


if __name__ == "__main__": 
    pass