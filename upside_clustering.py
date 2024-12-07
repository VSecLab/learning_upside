import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

matplotlib.use('agg')

# TODO: add the plot to HTML 

def ml_kmeans_plot(dataset, sensor, kfit): 
    
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection="3d")

    distances = np.linalg.norm(dataset, axis=1) # euclidean distance of each point from the origin (for coloring)

    ax.scatter3D(
        dataset[:, 0], dataset[:, 1], dataset[:, 2], c=distances, alpha=0.3, cmap="viridis"
    )
    plt.title(sensor + " 3D data point and cluster centers")

    centers = kfit.cluster_centers_
    ax.scatter3D(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        alpha=1,
        c="red",
        marker="X",
    )

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()

def ml_kmeans(df, sensor):
    dataset = df[["varianceX", "varianceY", "varianceZ"]].to_numpy()
    #print(dataset)
    kmeans = KMeans(n_clusters=2)
    kfit = kmeans.fit(dataset)
    res = kmeans.predict(dataset)

    ml_kmeans_plot(dataset, sensor, kfit)
    
    #print(res)

    new_df = df
    new_df = new_df.reset_index()
    new_df.insert(6, "motion", res, allow_duplicates=True)

    new_df_1 = new_df[new_df["motion"] == 1]
    new_df_0 = new_df[new_df["motion"] == 0]

    #print(new_df_1)
    #print(new_df_0)

    return new_df_0, new_df_1


if __name__ == "__main__": 
    pass