import math
from abc import ABC, abstractmethod

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from algorithms.const import NUMBER_OF_TEAMS, PLAYERS_PER_TEAM, RANDOM_STATE, MAPS_PER_GAME, COUNTRY_SUFFIX
from data_preprocessing import PLAYERS_AVG_RATING_FILE_PATH
from algorithms.country_info_provider import CountryInfoProvider
import numpy as np


class BaseAlgorithm(ABC):

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.scaler = StandardScaler()
        self.use_country_data = False
        self.country_info_provider = None

    def with_country(self, use_players_country=False, use_avg_country_rating=False,
                     use_commonness=False, use_players_country_percentage=False,
                     use_avg_team_country_rating=False):
        if use_players_country == True or use_avg_country_rating == True or use_players_country_percentage == True:
            self.use_country_data = True

        self.country_info_provider = CountryInfoProvider(use_players_country, use_avg_country_rating,
                                                             use_commonness, use_players_country_percentage,
                                                             use_avg_team_country_rating)
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
        avg_country_rating = []

        if self.country_info_provider is not None:
            if self.country_info_provider.use_commonness:
               country_commonness += self.country_info_provider.get_commonness_of_team_country(match_data)
            if self.country_info_provider.use_avg_team_country_rating:
                avg_country_rating += self.country_info_provider.get_avg_team_country_rating(match_data)

        for t_idx in range(1, NUMBER_OF_TEAMS + 1):
            for p_idx in range(1, PLAYERS_PER_TEAM + 1):
                data_of_interest += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}'],
                                     match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_rating"}']]

                if self.use_country_data:
                    if self.country_info_provider.use_players_country:
                        country_data += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country"}']]
                    if self.country_info_provider.use_avg_country_rating:
                        country_data += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country_rating"}']]
                    if self.country_info_provider.use_players_country_percentage:
                        country_data += [self.country_info_provider.get_country_percentage_in_dataset(
                            match_data[
                                f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country"}'])]

                player_names += [match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_name"}']]

        self.country_features = len(country_data) / 10
        return country_commonness + avg_country_rating + data_of_interest + player_names + country_data

    def predict(self):
        print("Started prediction")
        y_pred = self.model.predict(self.x_test)
        y_pred = self.round_predictions(y_pred)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, y_pred))

    @staticmethod
    def round_predictions(y_pred):
        return list(map(lambda a: 1 if abs(1 - a) < abs(0 - a) else 0, y_pred))

    @abstractmethod
    def fit(self):
        pass

    def remove_unused_rows(self):
        new_x_test = []
        new_x_train = []
        for index, val in enumerate(self.x_test):
            if self.use_country_data:
                a = np.concatenate((val[:int(-1 * (self.country_features * 10) - NUMBER_OF_TEAMS * PLAYERS_PER_TEAM)],
                                    val[-1 * NUMBER_OF_TEAMS * PLAYERS_PER_TEAM:]), axis=0)
                new_x_test.append(a)
            else:
                new_x_test.append(val[:-1 * NUMBER_OF_TEAMS * PLAYERS_PER_TEAM])

        for index, val in enumerate(self.x_train):
            if self.use_country_data:
                a = np.concatenate((val[:int(-1 * (self.country_features * 10) - NUMBER_OF_TEAMS * PLAYERS_PER_TEAM)],
                                    val[-1 * NUMBER_OF_TEAMS * PLAYERS_PER_TEAM:]), axis=0)
                new_x_train.append(a)
            else:
                new_x_train.append(val[:-1 * NUMBER_OF_TEAMS * PLAYERS_PER_TEAM])

        self.x_test = new_x_test
        self.x_train = new_x_train

    def replace_test_input_rating_with_avg(self):
        players_ratings = pd.read_csv(PLAYERS_AVG_RATING_FILE_PATH, delimiter=',', encoding="utf8")
        ratings_dict = dict(players_ratings.values)
        country_features_ofset = int(-1 * ((self.country_features + 1) * 10))
        rating_offset = 0
        if self.country_info_provider is not None:
            if self.country_info_provider.use_commonness:
                rating_offset += 2
            if self.country_info_provider.use_avg_team_country_rating:
                rating_offset += 2

        print(rating_offset, country_features_ofset)
        for row in self.x_test:
            row[6 + rating_offset] = ratings_dict[row[-10 if not self.use_country_data else country_features_ofset]]
            row[8 + rating_offset] = ratings_dict[row[-9 if not self.use_country_data else country_features_ofset + 1]]
            row[10 + rating_offset] = ratings_dict[row[-8 if not self.use_country_data else country_features_ofset + 2]]
            row[12 + rating_offset] = ratings_dict[row[-7 if not self.use_country_data else country_features_ofset + 3]]
            row[14 + rating_offset] = ratings_dict[row[-6 if not self.use_country_data else country_features_ofset + 4]]
            row[16 + rating_offset] = ratings_dict[row[-5 if not self.use_country_data else country_features_ofset + 5]]
            row[18 + rating_offset] = ratings_dict[row[-4 if not self.use_country_data else country_features_ofset + 6]]
            row[20 + rating_offset] = ratings_dict[row[-3 if not self.use_country_data else country_features_ofset + 7]]
            row[22 + rating_offset] = ratings_dict[row[-2 if not self.use_country_data else country_features_ofset + 8]]
            row[24 + rating_offset] = ratings_dict[row[-1 if not self.use_country_data else country_features_ofset + 9]]
