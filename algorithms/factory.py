from algorithms.extreme_gradient_boosting import ExtremeGradientBoostingAlgorithm
from algorithms.random_forest import RandomForestRegressorAlgorithm


class AlgorithmFactory:

    @staticmethod
    def create(algorithm_name, data_path_or_data):
        if not algorithm_name in AlgorithmFactory.switcher:
            raise NotImplementedError("Not implemented for algorithm: " + algorithm_name)

        return AlgorithmFactory.switcher[algorithm_name](data_path_or_data)

    @staticmethod
    def create_extreme_gradient_boosting_regressor(data_path_or_data):
        parameters = {
            'param': {
                'max_depth': 5,
                'eta': 1,
                'objective': 'binary:logistic',
                'nthread': 4,
                'eval_metric': 'auc'
            },
            'num_round': 10,
            'early_stopping_rounds': 10,
        }

        return ExtremeGradientBoostingAlgorithm(data_path_or_data, parameters)

    @staticmethod
    def create_random_forest_regressor(data_path_or_data):
        parameters = {
            "rf__n_estimators": 500,
            "rf__max_features": 'auto',
            "rf__max_depth": 15,
            "rf__min_samples_leaf": 1,
            "rf__min_samples_split": 2,
            'rf__criterion': 'squared_error',
            'verbose': 1
        }

        return RandomForestRegressorAlgorithm(data_path_or_data, parameters)

    @staticmethod
    def get_algorithm_names():
        algorithm_names = list(AlgorithmFactory.switcher.keys())
        return algorithm_names

    switcher = {
        "XG_BOOST_REGRESSOR": create_extreme_gradient_boosting_regressor.__func__,
        "RANDOM_FOREST_REGRESSOR": create_random_forest_regressor.__func__,
    }
