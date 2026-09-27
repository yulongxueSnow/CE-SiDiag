import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib
from numpy import vectorize
from tools import model_visualization

# Define chaos map function (Logistic map)
def logistic_map(x, r):
    return r * x * (1 - x)

# Define chaos map function (Tent map)
def tent_map(x, r):
    if x < 0.5:
        return 1/r * x
    else:
        return 1/(1-r) * (1 - x)

# Define chaos map function (Sine map)
def sine_map(x, r):
    return r * np.sin(np.pi * x)

# Vectorized chaos map functions
vectorized_logistic_map = np.vectorize(logistic_map)
vectorized_tent_map = np.vectorize(tent_map)
vectorized_sine_map = np.vectorize(sine_map)

# Define chaos model classifier
class ChaosModel(ClassifierMixin, BaseEstimator):
    def __init__(self, chaos_iterations=10, r=3.5, chaos_map='logistic'):
        self.chaos_iterations = chaos_iterations
        self.r = r
        self.chaos_map = chaos_map
        self.classifier = GaussianNB()

    # Generate chaos features
    def generate_chaos_features(self, X):
        n_samples, n_features = X.shape
        chaos_features = np.zeros((n_samples, n_features), dtype=float)  # Initialize 2D array
        for i in range(n_features):  # Iterate over each column
            feature = X[:, i]  # Extract values of column i
            # Add small noise
            noise = np.random.normal(0, 0.01, size=feature.shape)
            chaos_feature = feature + noise
            chaos_feature = np.clip(chaos_feature, 0, 1)  # Clip initial values to [0, 1] range

            for _ in range(self.chaos_iterations):
                if self.chaos_map == 'logistic':
                    chaos_feature = vectorized_logistic_map(chaos_feature, self.r)
                elif self.chaos_map == 'tent':
                    chaos_feature = vectorized_tent_map(chaos_feature, self.r)
                elif self.chaos_map == 'sine':
                    chaos_feature = vectorized_sine_map(chaos_feature, self.r)
                else:
                    raise ValueError("Unsupported chaos map: %s" % self.chaos_map)
                chaos_feature = np.clip(chaos_feature, 0, 1)  # Clip values to [0, 1] range

            chaos_features[:, i] = chaos_feature  # Assign processed feature to the corresponding column

        return chaos_features

    # Train the chaos model
    def fit(self, X, y):
        chaos_X = self.generate_chaos_features(X)
        # model_visualization.visualize_chaos_features(X, chaos_X, y)
        self.classifier.fit(chaos_X, y)
        return self

    # Predict
    def predict(self, X):
        chaos_X = self.generate_chaos_features(X)
        return self.classifier.predict(chaos_X)

class ChaosEnsembleModel:
    def __init__(self, data_fpath):
        self.data_fpath = data_fpath
        self.scaler = StandardScaler()
        
    def load_data(self):
        df = pd.read_csv(self.data_fpath)
        X = df.drop('label', axis=1).values
        y = df['label'].values
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        
        # Feature scaling
        self.train_x_scaled = self.scaler.fit_transform(train_x)
        self.test_x_scaled = self.scaler.transform(test_x)
        return self.train_x_scaled, self.test_x_scaled, train_y, test_y, df.columns

    def train_model(self, chaos_map):
        # Create base classifier
        self.chaos_model = ChaosModel(chaos_iterations=chaos_map['chaos_iterations'], r=chaos_map['r'], chaos_map=chaos_map['chaos_map'])
        
        # Create ensemble model
        self.ensemble_model = VotingClassifier(estimators=[
            ('chaos_nb', self.chaos_model),
            ('svm', SVC()),
            ('knn', KNeighborsClassifier())
        ])
        self.ensemble_model.fit(self.train_x_scaled, self.train_y)
        return self.ensemble_model

    def evaluate(self, test_x, test_y):
        y_pred = self.ensemble_model.predict(test_x)
        return accuracy_score(test_y, y_pred) * 100

    def proba_data(self, test_x, test_y):
        y_predict = self.ensemble_model.predict(test_x)         # Return predicted labels (class) for the test samples
        accuracy = accuracy_score(test_y, y_predict) * 100  # Compute the accuracy of the predictions
        # print(f"Chaos Map:{self.chaos_map} , Accuracy: {accuracy:.2f}%")
        return y_predict, accuracy

    def exc_p(self, model_prefix,chaos_map):
        self.chaos_map = chaos_map
        self.train_x, self.test_x, self.train_y, self.test_y, df_columns = self.load_data()
        clf = self.train_model(chaos_map=chaos_map)  # Use default sine map
        y_predict, accuracy = self.proba_data(self.test_x, self.test_y)

        # Save model results
        joblib.dump(clf, './data/%s_model.pkl'%model_prefix)

        return self.train_x, self.test_x, self.train_y, self.test_y, y_predict, accuracy, df_columns

if __name__ == '__main__':
    n_components = 13
    data_fpaths = [
        './data/ib600_2_features_pca%d.csv' % n_components,
        './data/tb600_2_features_pca%d.csv' % n_components,
        './data/ob600_2_features_pca%d.csv' % n_components
    ]

    for data_fpath in data_fpaths:
        model_prefix = 'ib' if 'ib' in data_fpath else 'tb' if 'tb' in data_fpath else 'ob'
        print(f"Processing data: {data_fpath}, Model: {model_prefix}")

        chaos_maps = [
            # {'chaos_map': 'logistic','chaos_iterations':10,'r':4},
            # {'chaos_map': 'tent','chaos_iterations':10,'r':0.5},
            {'chaos_map': 'sine','chaos_iterations':10,'r':0.99}

        ]
        for chaos_map in chaos_maps:
        
            model = ChaosEnsembleModel(data_fpath)
            model.exc_p(model_prefix,chaos_map)