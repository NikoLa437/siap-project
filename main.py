import csv
import pandas as pd
from data_preprocessing import data_set_processing, FINAL_DATASET_WITH_COUNTRY_FILE_PATH, FINAL_DATASET_FILE_PATH, average_k_means_data_for_players

#from random_forest import RandomForestRegressorAlgorithm
#from extreme_gradient_boosting import ExtremeGradientBoostingAlgorithm

#from algorithms.factory import AlgorithmFactory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


class PlayerBasicStatistic:
    def __init__(self, playerCountry, avgNumberOfKils, avgNumberOfAssits, avgNumberOfDeath, avgMatchRating):
        self.playerCountry = playerCountry
        self.avgNumberOfKils = avgNumberOfKils
        self.avgNumberOfAssits = avgNumberOfAssits
        self.avgNumberOfDeath = avgNumberOfDeath
        self.avgMatchRating = avgMatchRating

    def __str__(self):
        return "Player: " + self.playerCountry + "\nAverage kills: " + str(self.avgNumberOfKils) + "\nAverage assists: " + str(self.avgNumberOfAssits) + "\nAverage deaths: " + str(self.avgNumberOfDeath) + "\nAverage match 2.0 rating: " + str(self.avgMatchRating)
    
# Indeksi za pristup podacima vezanih za igraca
# 1 - Naziv igraca, 2 - Tim, 3 - Protivnik, 4- Drzava, 9- broj mapa koje su igrane
# 13 - Broj kilova igraca, 14 - Broj asistencija, 15 - Broj smrti, 22 - 2.0 rating u mecu
# 10 - Mapa1, 23 - mapa1 kills, 24 - mapa1 assists, 25 - mapa 1 deaths, 26 - mapa1 hs, 32 - mapa1 rating 2.0
# 11 - Mapa2, 33 - mapa2 kills, 34 - mapa2 assists, 35 -mapa2 deaths, 36 - mapa2 hs, 42 - mapa2 rating 2.0
# 12 - Mapa3, 43 - mapa3 kills, 44 - mapa3 asissts, 45 - mapa3 deaths, 46 -mapa3 hs, 52 - mapa3 rating 2.0
# 16 - Headshots, , 23 - mapa1 kills, 24 mapa1 deaths
# 53 - kill in CT, 54 - death in CT, 58 - rating 2.0 in CT
# 59 - kill in TT, 60 - death in TT, 64 - rating 2.0 in TT

def readStatisticsForPlayer(playerName):
    playerCountry = ''
    sumNumberOfKils = 0
    sumNumberOfAssits = 0
    sumNumberOfDeath = 0
    sumMatchRating = 0.0
    numberOfRecords = 0

    with open('datasets/players.csv', newline='', encoding="utf8") as csvfile:
        players = csv.reader(csvfile, delimiter=',', quotechar='|')
        for playerData in players:
            if playerData[1].lower()==playerName.lower():
                #print(playerData[1] + '-' + playerData[4] +'-'+ playerData[13]+'-'+playerData[14]+'-'+playerData[15]+'-'+playerData[22])
                numberOfRecords= numberOfRecords+1
                playerCountry= playerData[4]
                sumNumberOfKils= sumNumberOfKils + int(playerData[13])
                sumNumberOfAssits= sumNumberOfAssits + int(playerData[14])
                sumNumberOfDeath= sumNumberOfDeath + int(playerData[15])
                sumMatchRating= sumMatchRating + float(playerData[22])

    return PlayerBasicStatistic(playerCountry, sumNumberOfKils/numberOfRecords, sumNumberOfAssits/numberOfRecords, sumNumberOfDeath/numberOfRecords, sumMatchRating/numberOfRecords)

PLAYERS_AVG_K_MEANS_DATA_FILE_PATH= 'datasets/players_avg_kmeans_data.csv'


def load_data():
    kmeans_data = pd.read_csv(PLAYERS_AVG_K_MEANS_DATA_FILE_PATH)

    kmeans_data[kmeans_data.columns] = StandardScaler().fit_transform(kmeans_data)

    return kmeans_data

def pca_embeddings(df_scaled):
    """To reduce the dimensions of the wine dataset we use Principal Component Analysis (PCA).
    Here we reduce it from 13 dimensions to 2.
    :param df_scaled: scaled data
    :return: pca result, pca for plotting graph
    """

    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(df_scaled)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))
    return pca_2_result, pca_2

def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.
    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5,6,7,8,9, 10]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()

    return best_grid['n_clusters']

def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters
    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    plt.scatter(x, y, c=label, alpha=0.5, s= 200)  # plot different colors per cluster
    plt.title('Player clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5)

    plt.show()

def main():
    #data_set_processing()
    #average_k_means_data_for_players()
    scaled_data = load_data()
    pca_result, pca_2 = pca_embeddings(scaled_data)
    #optimum_num_clusters = kmean_hyper_param_tuning(scaled_data)
    #print("optimum num of clusters =", optimum_num_clusters)
    #print('prosao')

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(scaled_data)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_2.transform(centroids)

    print(kmeans.cluster_centers_)

    print("4. Visualizing the data")
    visualizing_results(pca_result, kmeans.labels_, centroids_pca)
    #team1 = input('Unesi prvi tim: ')
    #team2 = input('Unesi drugi tim: ')
    #team1players = [] 
    #team2players = [] 

    #for i in range(1):
    #    team1players.append(input(f'Unesi {i+1}. igraca iz ' + team1 +': '))

    #for i in range(1):
    #    team2players.append(input(f'Unesi {i+1}. igraca iz ' + team2 +': '))

    #playerStatisticsFirstTeam = []
    #playerStatisticsSecondTeam = []

    #for playerName in team1players:
    #    playerStatisticsFirstTeam.append(readStatisticsForPlayer(playerName))

    #for playerName in team2players:
    #    playerStatisticsSecondTeam.append(readStatisticsForPlayer(playerName))

    #print('Za tim: ' + team1 + ' igraju sledeci igraci sa statistikama: \n')
    #for playerWithStatistics in playerStatisticsFirstTeam:
    #    print(playerWithStatistics)

    #print('Za tim: ' + team2 + ' igraju sledeci igraci sa statistikama: \n')
    #for playerWithStatistics in playerStatisticsSecondTeam:
    #    print(playerWithStatistics)

if __name__ == "__main__":
    main()
    # average_ranking_for_players()
    # convert_country_to_num()
    # convert_country_to_num()
    # rfalg = RandomForestRegressorAlgorithm('datasets/final.csv')#.with_country()
    # rfalg.load_data()
    # rfalg.fit()
    # rfalg.predict()

    #random_forest_alg = AlgorithmFactory.create(AlgorithmFactory.get_algorithm_names()[0], FINAL_DATASET_FILE_PATH)#.with_country()
    #random_forest_alg.load_data()
    #random_forest_alg.fit()
    #random_forest_alg.predict()

    # bst = ExtremeGradientBoostingAlgorithm('datasets/final.csv')
    # bst.train()
    # bst.predict()