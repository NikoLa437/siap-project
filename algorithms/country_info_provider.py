import math

import pandas as pd

from algorithms.const import PLAYERS_PER_TEAM, NUMBER_OF_TEAMS, COUNTRY_SUFFIX
from data_preprocessing import PLAYERS_WITH_COUNTRIES_FILE_PATH, get_country_percentage_in_dataset


class CountryInfoProvider:

    def __init__(self, use_players_country=False, use_avg_country_rating=False,
                 use_commonness=False, use_players_country_percentage=False,
                 use_avg_team_country_rating=False):

        if use_players_country_percentage:
            country_df = pd.read_csv(PLAYERS_WITH_COUNTRIES_FILE_PATH, encoding='utf-8')
            self.country_df_len = len(country_df)
            self.count_countries = dict(country_df['country_number'].value_counts())

        self.use_players_country = use_players_country
        self.use_avg_country_rating = use_avg_country_rating
        self.use_commonness = use_commonness
        self.use_players_country_percentage = use_players_country_percentage
        self.use_avg_team_country_rating = use_avg_team_country_rating

    def get_commonness_of_team_country(self, match_data):
        most_common_team_1 = self.get_commonness_metric(list(self.get_keys_by_suffix(1, match_data).values()))
        most_common_team_2 = self.get_commonness_metric(list(self.get_keys_by_suffix(2, match_data).values()))
        return [most_common_team_1, most_common_team_2]

    @staticmethod
    def get_keys_by_suffix(team_num, match_data):
        return {key: val for key, val in match_data.items() if key.endswith(f'{team_num}{"_"}{COUNTRY_SUFFIX}')}

    @staticmethod
    def get_commonness_metric(data_list):
        metric = 0
        for item in set(data_list):
            metric += math.pow(data_list.count(item), 2)

        return metric / (PLAYERS_PER_TEAM * PLAYERS_PER_TEAM)

    @staticmethod
    def get_avg_team_country_rating(match_data):
        rating = [0, 0]
        for t_idx in range(1, NUMBER_OF_TEAMS + 1):
            for p_idx in range(1, PLAYERS_PER_TEAM + 1):
                rating[t_idx - 1] += match_data[f'{"player_"}{p_idx}{"_team_"}{t_idx}{"_country_rating"}']

        return [rating[0] / PLAYERS_PER_TEAM, rating[1] / PLAYERS_PER_TEAM]

    def get_country_percentage_in_dataset(self, country):
        return get_country_percentage_in_dataset(self.count_countries, self.country_df_len, country)