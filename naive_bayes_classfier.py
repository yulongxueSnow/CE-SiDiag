#! Gaussian Naive Bayes classifier
import pdb
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from test_results import *
import joblib
 
# Define the Naive Bayes classifier
class BayesModel():
    def __init__(self,data_fpath):
        self.data_fpath = data_fpath
    
    # Load the data
    def load_data(self,model_prefix):
        df = pd.read_csv(self.data_fpath)
        X = df.drop('label', axis=1)  # Feature data, excluding the 'label' column
        y = df['label']  # Target variable, i.e., the 'label' column
        feature_names = df.columns[:-1]  # Feature names, excluding the 'label' column

        # Initialize the MinMaxScaler object
        # scaler = MinMaxScaler()

        # Apply min-max scaling to the data
        # X_scaled = scaler.fit_transform(X)

        # Save the normalization result
        # joblib.dump(scaler, './data/%s_bayes_scaler_obj.pkl'%model_prefix)
        
        features = pd.DataFrame(data=X, columns=feature_names)
        train_x, test_x, train_y, test_y = train_test_split(features, y, test_size=0.2)

        # print("Training data class distribution:\n", train_y.value_counts())
        # print("Test data class distribution:\n", test_y.value_counts())
 
        return train_x, test_x, train_y, test_y,df.columns
    
    # Train the Gaussian Naive Bayes model
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    
    # Process the prediction data
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)         # Return the predicted class of the samples to be predicted
        y_proba = clf.predict_proba(test_x)     # Return the probability of the predicted samples belonging to each label
        accuracy = accuracy_score(test_y, y_predict) * 100  # Calculate the accuracy of the predictions
        
        return y_predict, y_proba, accuracy
    
    # Training data
    def exc_p(self,model_prefix=''):
        train_x, test_x, train_y, test_y, df_columns = self.load_data(model_prefix)     # Load the data
        clf = self.train_model(train_x, train_y)    # Train the Gaussian Naive Bayes model clf
        y_predict, y_proba, accuracy = self.proba_data(clf, test_x, test_y)  # Use the trained model clf to predict the test set test_x

        # Save the model result
        joblib.dump(clf, './data/%s_bayes_model.pkl'%model_prefix)
 
        return train_x, test_x, train_y, test_y, y_predict, y_proba, accuracy, df_columns,clf
        
 
if __name__ == '__main__':

    n_components=13

    data_fpaths=['./data/ib600_2_features_pca%d.csv'%n_components,
                './data/tb600_2_features_pca%d.csv'%n_components,
                './data/ob600_2_features_pca%d.csv'%n_components]
    for data_fpath in data_fpaths:
        model_obj = BayesModel(data_fpath)
        if 'ib' in data_fpath:
            model_prefix='ib'
        elif 'tb' in data_fpath:
            model_prefix='tb'
        elif 'ob' in data_fpath:
            model_prefix='ob'
        else:
            model_prefix=''
        train_x, test_x, train_y, test_y, y_predict, y_proba, accuracy, df_columns,clf = model_obj.exc_p(model_prefix)

        show_results(test_y,y_predict)