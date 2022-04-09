from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import pandas as pd

class KmeansAlgorithm:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.scaler = StandardScaler()
        self.kmeans_data = []
        self.kmeans = None
        self.pca_results = []
        self.pca = None
    
    def load_data(self):
        loaded_data = pd.read_csv(self.path_to_data)

        loaded_data = loaded_data.drop(loaded_data.columns[0], axis=1)
    
        loaded_data[loaded_data.columns] = StandardScaler().fit_transform(loaded_data)

        self.kmeans_data = loaded_data

    def pca_embeddings(self):
        """Here we reduce it from 13 dimensions to 2.
        :param df_scaled: scaled data
        :return: pca result, pca for plotting graph
        """

        pca_2 = PCA(n_components=2)
        pca_2_result = pca_2.fit_transform(self.kmeans_data)

        self.pca_results=  pca_2_result
        self.pca = pca_2

    def fit(self, clusters):
        self.load_data()
        self.kmeans = KMeans(n_clusters=clusters)
        self.kmeans.fit(self.kmeans_data)

        centroids = self.kmeans.cluster_centers_

        print(centroids)

    def visualizing_results(self):
        self.pca_embeddings() # embeddings data for vizualization
        x = self.pca_results[:, 0]
        y = self.pca_results[:, 1]
        centroids = self.kmeans.cluster_centers_

        plt.scatter(x, y, c=self.kmeans.labels_, alpha=0.5, s= 200)  # plot different colors per cluster
        plt.title('Player clusters')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        plt.scatter(self.pca.transform(centroids)[:, 0], self.pca.transform(centroids)[:, 1], marker='X', s=200, linewidths=1.5,
                    color='red', edgecolors="black", lw=1.5)

        plt.show()

    def predict(self, data):
        return self.kmeans.predict(data)

    def predict_and_save_to_file_player_clusters(self, players_kmeans_file, save_file_path):
        players_avg_kmeans_data = pd.read_csv(players_kmeans_file,  encoding='utf-8')

        scaled_features = players_avg_kmeans_data.copy()
        col_names = ['kills', 'assists', 'deaths', 'hs', 'rating']
        features = scaled_features[col_names]

        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)

        scaled_features[col_names] = features

        dictsArray = []
        for i, row in scaled_features.iterrows():
            player_dict={}

            player_dict['player_name']=row['player_name']
           
            player_data = [float(row['kills']),float(row['assists']),float(row['deaths']),float(row['hs']),float(row['rating'])]
            
            print(i)
            player_dict['cluster']= self.kmeans.predict([[float(row['kills']),float(row['assists']),float(row['deaths']),float(row['hs']),float(row['rating'])]])[0]

            dictsArray.append(player_dict)

        
        df = pd.DataFrame.from_dict(dictsArray) 

        df.to_csv(save_file_path, index = False, header=True)
