import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# computes a distance matrix between data points (full matrix)
def custom_distance_matrix(data, metric='euclidean'):
    n = len(data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix

# main function  
def custom_dbscan(normalized_distance, e, k):
    DistanceMatrix = custom_distance_matrix(normalized_distance, metric='euclidean')
    core_point_array = np.zeros(len(normalized_distance))
    cluster_array = np.zeros(len(normalized_distance))
    # cluster label
    w = 0

    for i in range(len(DistanceMatrix)):
        PointNeighbors = np.where(DistanceMatrix[i] <= e)[0]
        if len(PointNeighbors) >= k:
            core_point_array[i] = 1  # marked as a core point 
            # If the point has not been assigned to any cluster, it is assigned to a new cluster 
            if cluster_array[i] == 0:
                cluster_array[i] = w
                w += 1
            for x in range(len(PointNeighbors)):
                if cluster_array[PointNeighbors[x]] == 0:
                    cluster_array[PointNeighbors[x]] = cluster_array[i]

    return cluster_array

# data preparation 
def prepare_data(input_data):
    # create higher-order features to capture more complex relationships between features
    poly = PolynomialFeatures(4)
    input_data = poly.fit_transform(input_data)
    #  transform the distribution of each feature to be approximately uniform (mitigate impact of outliers)
    input_data = QuantileTransformer(n_quantiles=40, random_state=0).fit_transform(input_data)

    scaler = MinMaxScaler()
    scaler.fit(input_data)
    normalized_input_data = scaler.transform(input_data)
   
    distan = custom_distance_matrix(normalized_input_data, metric='euclidean')

    #  ensure that all distances have a consistent scale + ensures that all features contribute equally
    scaler.fit(distan)
    normalized_distance = scaler.transform(distan)
    #  capture the most important information 
    pca = PCA(n_components=4)
    normalized_distance = pca.fit_transform(normalized_distance)

    scaler.fit(normalized_distance)
    normalized_distance = scaler.transform(normalized_distance)

    return normalized_distance

def main():
    dataset = pd.read_csv("data1.csv")

    # Extract features and target
    input_data = dataset.drop(columns=['Fertility'])
    target_data = dataset['Fertility']

    # Data Manipulations before introducing to the algorithm
    normalized_distance = prepare_data(input_data)
    e = 0.5
    k = 40
    cluster_array = custom_dbscan(normalized_distance, e, k)
    num_clusters = len(np.unique(cluster_array))

    if num_clusters > 1:
    # Calculate silhouette score
     silhouette_avg = silhouette_score(normalized_distance, cluster_array)
     print('Silhouette Score:', silhouette_avg)
    else:
       print("Il y a moins de deux clusters.")

    plt.subplot(2, 1, 1)
    plt.scatter(normalized_distance[:, 0], normalized_distance[:, 1], c=cluster_array, cmap='Paired')
    plt.title("Custom DBSCAN Predicted Cluster Outputs")

    plt.subplot(2, 1, 2)
    plt.scatter(normalized_distance[:, 0], normalized_distance[:, 1], c=target_data, cmap='Paired')
    plt.title("Actual Target Outputs")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
