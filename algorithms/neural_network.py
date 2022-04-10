from keras.legacy_tf_layers.core import Dense
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from algorithms.algorithm import BaseAlgorithm

import numpy as np


class NeuralNetwork(BaseAlgorithm):
    def __init__(self, path_to_data):
        super().__init__(path_to_data)
        self.nnetwork = self.create_neural_network()

    def create_neural_network(self):
        nnetwork = Sequential()
        nnetwork.add(Dense(60, input_dim=37 if self.use_country_data else 12, activation='relu'))
        nnetwork.add(Dense(1, activation='sigmoid'))
        return nnetwork

    def fit(self):
        # za binarnu binary_crossentropy
        self.nnetwork.compile(loss='binary_crossentropy', optimizer='adam')
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        # obucavanje neuronske mreze
        self.nnetwork.fit(np.asarray(self.x_train), np.asarray(self.y_train), epochs=30,
                          batch_size=5, verbose=1, shuffle=False)

    def predict(self):
        y_train_pred = self.nnetwork.predict(self.x_train)
        y_test_pred = self.nnetwork.predict(self.x_test)
        print("Train accuracy: ", accuracy_score(self.y_train, y_train_pred.round()))
        print("Validation accuracy: ", accuracy_score(self.y_test, y_test_pred.round()))