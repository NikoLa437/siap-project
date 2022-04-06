import math

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import PLAYERS_AVG_RATING_FILE_PATH
import numpy as np

random_state = 42


class RandomForestRegressorAlgorithm():
    def __init__(self, path_to_data):

        self.path_to_data = path_to_data
        parameters = {
            "rf__n_estimators": 500,
            "rf__max_features": 'auto',
            "rf__max_depth": 15,
            "rf__min_samples_leaf": 1,
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
        self.use_country_data = False

    def with_country(self):
        self.use_country_data = True
        return self

    def load_data(self):
        matches = pd.read_csv(self.path_to_data, delimiter=',', encoding="utf8")
        matches = matches.reset_index()
        x = []
        y = []

        for index, match_data in matches.iterrows():
            for z in range(1, 4, 1):
                if self.chech_if_valid_row(match_data, f'{"map_"}{z}'):
                    train_data = [match_data['team_1'], match_data['team_2'], match_data['team1_rank'],
                                  match_data['team2_rank'],
                                  match_data[f'{"map_"}{z}'], match_data['player_1_team_1'],
                                  match_data['player_1_team_1_rating'],
                                  match_data['player_2_team_1'], match_data['player_2_team_1_rating'],
                                  match_data['player_3_team_1'], match_data['player_3_team_1_rating'],
                                  match_data['player_4_team_1'], match_data['player_4_team_1_rating'],
                                  match_data['player_5_team_1'], match_data['player_5_team_1_rating'],
                                  match_data['player_1_team_2'], match_data['player_1_team_2_rating'],
                                  match_data['player_2_team_2'], match_data['player_2_team_2_rating'],
                                  match_data['player_3_team_2'], match_data['player_3_team_2_rating'],
                                  match_data['player_4_team_2'], match_data['player_4_team_2_rating'],
                                  match_data['player_5_team_2'], match_data['player_5_team_2_rating'],
                                  match_data['player_1_team_1_name'],
                                  match_data['player_2_team_1_name'],
                                  match_data['player_3_team_1_name'],
                                  match_data['player_4_team_1_name'],
                                  match_data['player_5_team_1_name'],
                                  match_data['player_1_team_2_name'],
                                  match_data['player_2_team_2_name'],
                                  match_data['player_3_team_2_name'],
                                  match_data['player_4_team_2_name'],
                                  match_data['player_5_team_2_name'], ]

                    if self.use_country_data:
                        train_data += self.get_country_data(match_data)

                    x.append(train_data)
                    y.append(match_data[f'{"map_"}{z}{"_winner"}'])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(x), y, test_size=0.2,
                                                                                random_state=random_state)

        self.replace_test_input_rating_with_avg()
        self.remove_unused_rows()
       # print(self.x_test)

    def remove_unused_rows(self):
        new_x_test = []
        new_x_train = []
        for index, val in enumerate(self.x_test):
            new_x_test.append(val[0:-10 if not self.use_country_data else -20])

        for index, val in enumerate(self.x_train):
            new_x_train.append(val[0:-10 if not self.use_country_data else -20])

        self.x_test = new_x_test
        self.x_train = new_x_train


    def replace_test_input_rating_with_avg(self):
        players_ratings = pd.read_csv(PLAYERS_AVG_RATING_FILE_PATH, delimiter=',', encoding="utf8")
        ratings_dict = dict(players_ratings.values)
        for row in self.x_test:
            row[6] = ratings_dict[row[-10 if not self.use_country_data else -20]]
            row[8] = ratings_dict[row[-9 if not self.use_country_data else -19]]
            row[10] = ratings_dict[row[-8 if not self.use_country_data else -18]]
            row[12] = ratings_dict[row[-7 if not self.use_country_data else -17]]
            row[14] = ratings_dict[row[-6 if not self.use_country_data else -16]]
            row[16] = ratings_dict[row[-5 if not self.use_country_data else -15]]
            row[18] = ratings_dict[row[-4 if not self.use_country_data else -14]]
            row[20] = ratings_dict[row[-3 if not self.use_country_data else -13]]
            row[22] = ratings_dict[row[-2 if not self.use_country_data else -12]]
            row[24] = ratings_dict[row[-1 if not self.use_country_data else -11]]


    def get_country_data(self, match_data):
        return [match_data['player_1_team_1_country'], match_data['player_2_team_1_country'],
                match_data['player_3_team_1_country'], match_data['player_4_team_1_country'],
                match_data['player_5_team_1_country'], match_data['player_1_team_2_country'],
                match_data['player_2_team_2_country'], match_data['player_3_team_2_country'],
                match_data['player_4_team_2_country'], match_data['player_5_team_2_country']]

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

    def chech_if_valid_row(self, match_data, current_map):
        valid = self.check_standard_data(match_data, current_map)
        return valid and self.check_country_data(match_data)

    def check_standard_data(self, match_data, current_map):
        return not (math.isnan(match_data[current_map]) or math.isnan(match_data['team_1']) or math.isnan(
            match_data['team_2']) or math.isnan(match_data['team2_rank'])
                    or math.isnan(match_data['team1_rank']) or math.isnan(match_data['player_1_team_1']) or math.isnan(
                    match_data['player_1_team_1_rating'])
                    or math.isnan(match_data['player_2_team_1']) or math.isnan(
                    match_data['player_2_team_1_rating']) or math.isnan(
                    match_data['player_3_team_1'])
                    or math.isnan(match_data['player_3_team_1_rating']) or math.isnan(
                    match_data['player_4_team_1']) or math.isnan(
                    match_data['player_4_team_1_rating'])
                    or math.isnan(match_data['player_5_team_1']) or math.isnan(
                    match_data['player_5_team_1_rating']) or math.isnan(
                    match_data['player_1_team_2'])
                    or math.isnan(match_data['player_1_team_2_rating']) or math.isnan(
                    match_data['player_2_team_2']) or math.isnan(
                    match_data['player_2_team_2_rating'])
                    or math.isnan(match_data['player_3_team_2']) or math.isnan(
                    match_data['player_3_team_2_rating']) or math.isnan(
                    match_data['player_4_team_2'])
                    or math.isnan(match_data['player_4_team_2_rating']) or math.isnan(
                    match_data['player_5_team_2']) or math.isnan(
                     match_data['player_5_team_2_rating']) #or
                    # math.isnan(match_data['player_1_team_1_name']) or
                    # math.isnan(match_data['player_2_team_1_name']) or
                    # math.isnan(match_data['player_3_team_1_name']) or
                    # math.isnan(match_data['player_4_team_1_name']) or
                    # math.isnan(match_data['player_5_team_1_name']) or
                    # math.isnan(match_data['player_1_team_2_name']) or
                    # math.isnan(match_data['player_2_team_2_name']) or
                    # math.isnan(match_data['player_3_team_2_name']) or
                    # math.isnan(match_data['player_4_team_2_name']) or
                    # math.isnan(match_data['player_5_team_2_name'])
                    )

    def check_country_data(self, match_data):
        return not (math.isnan(match_data['player_1_team_1_country']) or math.isnan(
            match_data['player_2_team_1_country']) or math.isnan(match_data['player_3_team_1_country'])
                    or math.isnan(match_data['player_4_team_1_country']) or math.isnan(
                    match_data['player_5_team_1_country']) or math.isnan(match_data['player_1_team_2_country'])
                    or math.isnan(match_data['player_2_team_2_country']) or math.isnan(
                    match_data['player_3_team_2_country']) or math.isnan(match_data['player_4_team_2_country'])
                    or math.isnan(match_data['player_5_team_2_country'])) if self.use_country_data else True
