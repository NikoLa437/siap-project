from algorithms.algorithm import BaseAlgorithm
import xgboost as xgb


class ExtremeGradientBoostingAlgorithm(BaseAlgorithm):
    def __init__(self, path_to_data, parameters):
        super().__init__(path_to_data)

        self.parameters = parameters

    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        d_train = xgb.DMatrix(self.x_train, self.y_train)
        d_test = xgb.DMatrix(self.x_test, self.y_test)

        eval_list = [(d_test, 'eval'), (d_train, 'train')]

        self.model = xgb.train(self.parameters['param'], d_train, self.parameters['num_round'],
                               eval_list, early_stopping_rounds=self.parameters['early_stopping_rounds'])
        self.x_test = xgb.DMatrix(self.x_test)

