import math

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

random_state = 42

class RandomForestRegressorAlgorithm():
    def __init__(self, path_to_data):

        x, y = self.load_data(path_to_data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                random_state=random_state)
        parameters = {
            "rf__n_estimators": 600,
            "rf__max_features": 'auto',
            "rf__max_depth": 15,
            "rf__min_samples_leaf": 2,
            "rf__min_samples_split": 2,
            'rf__criterion': 'squared_error',
            'verbose': 1
        }

        self.scaler = StandardScaler()

        self.model = RandomForestRegressor(random_state=random_state,
                                           n_estimators=parameters['rf__n_estimators'],
                                           max_features=parameters['rf__max_features'],
                                           max_depth=parameters['rf__max_depth'],
                                           min_samples_split=parameters['rf__min_samples_split'],
                                           min_samples_leaf=parameters['rf__min_samples_leaf'],
                                           criterion=parameters['rf__criterion'],
                                           verbose=parameters['verbose']
                                           )

    def load_data(self, path_to_data):
        matches = pd.read_csv(path_to_data, delimiter=',', encoding="utf8")
        matches = matches.reset_index()
        x = []
        y = []

        for index, matchData in matches.iterrows():
            for z in range(1, 4, 1):
                if not self.chech_if_not_valid_row(matchData, f'{"map_"}{z}'):
                    x.append(
                        [matchData['team_1'], matchData['team_2'], matchData['team1_rank'], matchData['team2_rank'],
                         matchData[f'{"map_"}{z}'], matchData['player_1_team_1'], matchData['player_1_team_1_rating'],
                         matchData['player_1_team_1_country'],
                         matchData['player_2_team_1'], matchData['player_2_team_1_rating'],
                         matchData['player_2_team_1_country'],
                         matchData['player_3_team_1'], matchData['player_3_team_1_rating'],
                         matchData['player_3_team_1_country'],
                         matchData['player_4_team_1'], matchData['player_4_team_1_rating'],
                         matchData['player_4_team_1_country'],
                         matchData['player_5_team_1'], matchData['player_5_team_1_rating'],
                         matchData['player_5_team_1_country'],
                         matchData['player_1_team_2'], matchData['player_1_team_2_rating'],
                         matchData['player_1_team_2_country'],
                         matchData['player_2_team_2'], matchData['player_2_team_2_rating'],
                         matchData['player_2_team_2_country'],
                         matchData['player_3_team_2'], matchData['player_3_team_2_rating'],
                         matchData['player_3_team_2_country'],
                         matchData['player_4_team_2'], matchData['player_4_team_2_rating'],
                         matchData['player_4_team_2_country'],
                         matchData['player_5_team_2'], matchData['player_5_team_2_rating'],
                         matchData['player_5_team_2_country'],])
                    y.append(matchData[f'{"map_"}{z}{"_winner"}'])

        return np.array(x), y

    def fit(self):
        print("Started fitting")
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        print("Started prediction")
        y_pred = self.model.predict(self.x_test)
        y_pred = self.round_predictions(y_pred)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, y_pred))

    def round_predictions(self, y_pred):
        return list(map(lambda a: 2.0 if abs(2.0 - a) < abs(1.0 - a) else 1.0, y_pred))

    def chech_if_not_valid_row(self, match_data, current_map):
        return math.isnan(match_data[current_map]) or math.isnan(match_data['team_1']) or math.isnan(match_data['team_2']) or math.isnan(match_data['team2_rank'])\
               or math.isnan(match_data['team1_rank']) or math.isnan(match_data['player_1_team_1']) or math.isnan(match_data['player_1_team_1_rating'])\
               or math.isnan(match_data['player_2_team_1']) or math.isnan(match_data['player_2_team_1_rating']) or math.isnan(match_data['player_3_team_1'])\
               or math.isnan(match_data['player_3_team_1_rating']) or math.isnan(match_data['player_4_team_1']) or math.isnan(match_data['player_4_team_1_rating'])\
               or math.isnan(match_data['player_5_team_1']) or math.isnan(match_data['player_5_team_1_rating']) or math.isnan(match_data['player_1_team_2'])\
               or math.isnan(match_data['player_1_team_2_rating']) or math.isnan(match_data['player_2_team_2']) or math.isnan(match_data['player_2_team_2_rating'])\
               or math.isnan(match_data['player_3_team_2']) or math.isnan(match_data['player_3_team_2_rating']) or math.isnan(match_data['player_4_team_2'])\
               or math.isnan(match_data['player_4_team_2_rating']) or math.isnan(match_data['player_5_team_2']) or math.isnan(match_data['player_5_team_2_rating'])\
               or math.isnan(match_data['player_1_team_1_country']) or math.isnan(match_data['player_2_team_1_country']) or math.isnan(match_data['player_3_team_1_country']) \
               or math.isnan(match_data['player_4_team_1_country']) or math.isnan(match_data['player_5_team_1_country']) or math.isnan(match_data['player_1_team_2_country']) \
               or math.isnan(match_data['player_2_team_2_country']) or math.isnan(match_data['player_3_team_2_country']) or math.isnan(match_data['player_4_team_2_country']) \
               or math.isnan(match_data['player_5_team_2_country'])