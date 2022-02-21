import math

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb

random_state = 42

class ExtremeGradientBoostingAlgorithm():
    def __init__(self, path_to_data):

        x, y = self.load_data(path_to_data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                random_state=random_state)
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        self.dtrain = xgb.DMatrix(self.x_train, self.y_train)
        self.dtest = xgb.DMatrix(self.x_test, self.y_test)

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
                         matchData['player_2_team_1'], matchData['player_2_team_1_rating'],
                         matchData['player_3_team_1'], matchData['player_3_team_1_rating'],
                         matchData['player_4_team_1'], matchData['player_4_team_1_rating'],
                         matchData['player_5_team_1'], matchData['player_5_team_1_rating'],
                         matchData['player_1_team_2'], matchData['player_1_team_2_rating'],
                         matchData['player_2_team_2'], matchData['player_2_team_2_rating'],
                         matchData['player_3_team_2'], matchData['player_3_team_2_rating'],
                         matchData['player_4_team_2'], matchData['player_4_team_2_rating'],
                         matchData['player_5_team_2'], matchData['player_5_team_2_rating']])
                    y.append(matchData[f'{"map_"}{z}{"_winner"}'] - 1)

        return np.array(x), y

    def train(self):
        print("Started training")

        param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
        # binary:logistic: logistic regression for binary classification, output probability
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]
        num_round = 10

        self.model = xgb.train(param, self.dtrain, num_round, evallist, early_stopping_rounds=10)
        # print(self.model.best_score)
        # print(self.model.best_iteration)

    def predict(self):
        print("Started prediction")
        y_pred = self.model.predict(xgb.DMatrix(self.x_test))
        y_pred = self.round_predictions(y_pred)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, y_pred))

    def round_predictions(self, y_pred):
        return list(map(lambda a: 1 if abs(1 - a) < abs(0 - a) else 0, y_pred))

    def chech_if_not_valid_row(self, match_data, current_map):
        return math.isnan(match_data[current_map]) or math.isnan(match_data['team_1']) or math.isnan(match_data['team_2']) or math.isnan(match_data['team2_rank'])\
               or math.isnan(match_data['team1_rank']) or math.isnan(match_data['player_1_team_1']) or math.isnan(match_data['player_1_team_1_rating'])\
               or math.isnan(match_data['player_2_team_1']) or math.isnan(match_data['player_2_team_1_rating']) or math.isnan(match_data['player_3_team_1'])\
               or math.isnan(match_data['player_3_team_1_rating']) or math.isnan(match_data['player_4_team_1']) or math.isnan(match_data['player_4_team_1_rating'])\
               or math.isnan(match_data['player_5_team_1']) or math.isnan(match_data['player_5_team_1_rating']) or math.isnan(match_data['player_1_team_2'])\
               or math.isnan(match_data['player_1_team_2_rating']) or math.isnan(match_data['player_2_team_2']) or math.isnan(match_data['player_2_team_2_rating'])\
               or math.isnan(match_data['player_3_team_2']) or math.isnan(match_data['player_3_team_2_rating']) or math.isnan(match_data['player_4_team_2'])\
               or math.isnan(match_data['player_4_team_2_rating']) or math.isnan(match_data['player_5_team_2']) or math.isnan(match_data['player_5_team_2_rating'])