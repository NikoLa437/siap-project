from abc import abstractmethod

from sklearn.ensemble import RandomForestRegressor

from algorithms.algorithm import BaseAlgorithm, RANDOM_STATE


class RandomForestRegressorAlgorithm(BaseAlgorithm):
    def __init__(self, path_to_data, parameters):
        super().__init__(path_to_data)

        self.model = RandomForestRegressor(random_state=RANDOM_STATE,
                                           n_estimators=parameters['rf__n_estimators'],
                                           max_features=parameters['rf__max_features'],
                                           max_depth=parameters['rf__max_depth'],
                                           min_samples_split=parameters['rf__min_samples_split'],
                                           min_samples_leaf=parameters['rf__min_samples_leaf'],
                                           criterion=parameters['rf__criterion'],
                                           verbose=parameters['verbose']
                                           )

    def fit(self):
        print("Started fitting")
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model.fit(self.x_train, self.y_train)