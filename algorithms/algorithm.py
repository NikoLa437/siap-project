import math
from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import PLAYERS_AVG_RATING_FILE_PATH, PLAYERS_AVG_CUSTOM_RATING_FILE_PATH
import numpy as np

NUMBER_OF_TEAMS = 2
PLAYERS_PER_TEAM = 5
MAPS_PER_GAME = 3

RANDOM_STATE = 42

COUNTRY_SUFFIX = "country"


class BaseAlgorithm(ABC):

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.scaler = StandardScaler()
        self.use_country_data = False
        self.use_custom_rating = False

    def with_country(self):
        self.use_country_data = True
        return self

    def with_custom_rating(self):
        self.use_custom_rating = True
        return self

    def load_data(self):
        matches = pd.read_csv(self.path_to_data, delimiter=',', encoding="utf8")
        matches = matches.reset_index()
        x = []
        y = []

        for index, match_data in matches.iterrows():
            for z in range(1, MAPS_PER_GAME + 1, 1):
                if self.check_if_valid_row(match_data, f'{"map_"}{z}'):
                    train_set = [match_data['team_1'], match_data['team_2'], match_data['team1_rank'],
                                 match_data['team2_rank'], match_data[f'{"map_"}{z}']]

                    train_set += self.extract_player_data(match_data)

                    x.append(train_set)
                    y.append(match_data[f'{"map_"}{z}{"_winner"}'] - 1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(x), y, test_size=0.2,
                                                                                random_state=RANDOM_STATE)

        self.replace_test_input_rating_with_avg()
        self.remove_unused_rows()

    @staticmethod
    def check_if_valid_row(match_data, current_map):
        return not (math.isnan(match_data[current_map]) or math.isnan(match_data['team_1']) or math.isnan(
            match_data['team_2']) or math.isnan(match_data['team2_rank']) or math.isnan(match_data['team1_rank']))

    def extract_player_data(self, match_data):
        data_of_interest = []
        player_names = []
        country_data = []
        country_commonness = []
        if self.use_country_data:
            country_commonness += self.get_commonness_of_team_country(match_data)

        for t_idx in range(1, NUMBER_OF_TEAMS + 1):
            for p_idx in range(1, PLAYERS_PER_TEAM + 1):

                if self.use_custom_rating:
                    data_of_interest += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}'],
                                         match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_custom_rating"}']]
                else:
                    data_of_interest += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}'],
                                     match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_rating"}']]

                if self.use_country_data:
                    country_data += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country"}'],
                                     match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country_rating"}']]

                player_names += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_name"}']]



        return country_commonness + data_of_interest + player_names + country_data

    def predict(self):
        print("Started prediction")
        y_pred = self.model.predict(self.x_test)
        y_pred = self.round_predictions(y_pred)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, y_pred))

    @staticmethod
    def round_predictions(y_pred):
        return list(map(lambda a: 1 if abs(1 - a) < abs(0 - a) else 0, y_pred))

    def get_commonness_of_team_country(self, match_data):
        most_common_team_1 = self.max_repeat(list(self.get_keys_by_suffix(1, match_data).values()))
        most_common_team_2 = self.max_repeat(list(self.get_keys_by_suffix(2, match_data).values()))
        return [most_common_team_1 / PLAYERS_PER_TEAM, most_common_team_2 / PLAYERS_PER_TEAM]

    @staticmethod
    def max_repeat(data_list):
        items = max(set(data_list), key=data_list.count)
        return data_list.count(items)

    @staticmethod
    def get_keys_by_suffix(team_num, match_data):
        return {key: val for key, val in match_data.items() if key.endswith(f'{team_num}{"_"}{COUNTRY_SUFFIX}')}

    @abstractmethod
    def fit(self):
        pass

    def remove_unused_rows(self):
        new_x_test = []
        new_x_train = []
        for index, val in enumerate(self.x_test):
            new_x_test.append(val[0:-10 if not self.use_country_data else -30])

        for index, val in enumerate(self.x_train):
            new_x_train.append(val[0:-10 if not self.use_country_data else -30])

        self.x_test = new_x_test
        self.x_train = new_x_train

    def replace_test_input_rating_with_avg(self):
        if self.use_custom_rating:
            players_ratings = pd.read_csv(PLAYERS_AVG_CUSTOM_RATING_FILE_PATH, delimiter=',', encoding="utf8")
        else:
            players_ratings = pd.read_csv(PLAYERS_AVG_RATING_FILE_PATH, delimiter=',', encoding="utf8")
        ratings_dict = dict(players_ratings.values)
        for row in self.x_test:
            row[6] = ratings_dict[row[-10 if not self.use_country_data else -30]]
            row[8] = ratings_dict[row[-9 if not self.use_country_data else -29]]
            row[10] = ratings_dict[row[-8 if not self.use_country_data else -28]]
            row[12] = ratings_dict[row[-7 if not self.use_country_data else -27]]
            row[14] = ratings_dict[row[-6 if not self.use_country_data else -26]]
            row[16] = ratings_dict[row[-5 if not self.use_country_data else -25]]
            row[18] = ratings_dict[row[-4 if not self.use_country_data else -24]]
            row[20] = ratings_dict[row[-3 if not self.use_country_data else -23]]
            row[22] = ratings_dict[row[-2 if not self.use_country_data else -22]]
            row[24] = ratings_dict[row[-1 if not self.use_country_data else -21]]
