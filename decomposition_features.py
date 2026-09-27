import pdb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from test_results import *

def process_data(features_df,n_components=10,save_model_prefix_name=''):

    # Pearson correlation coefficient
    # Compute the Pearson correlation coefficient matrix across all features
    correlation_matrix = features_df.corr()

    # Set the correlation threshold; feature pairs above this value are considered highly correlated
    threshold = 0.8

    # Initialize an empty list to store the column names to drop
    columns_to_drop = []

    # Iterate over the upper triangle (the lower triangle is symmetric, no need to check twice)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            # If the absolute correlation between features exceeds the threshold
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                # Add one of the highly correlated features to the list (using j here; the selection strategy can be customized)
                columns_to_drop.append(correlation_matrix.columns[j])

    # Ensure not all columns are removed
    columns_to_drop = list(set(columns_to_drop))  # Remove duplicates while preserving order
    if len(columns_to_drop) == len(features_df.columns):
        print("Warning: all features are marked as highly correlated; this may be caused by a threshold set too low.")
    else:
        # Remove the highly correlated features from the original dataframe
        df_filtered = features_df.drop(columns=columns_to_drop)
        
        print("\nFiltered features:")
        print(df_filtered.head())  # Print the first few rows of the filtered dataframe as a check

    # Data standardization (optional, but important for PCA, especially when feature scales differ greatly)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Save the PCA normalization (scaler) model
    joblib.dump(scaler, './data/%s_pca_scaler.pkl'%save_model_prefix_name)   

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)

    # pca = PCA().fit(scaled_features)
    # # Compute the cumulative explained variance
    # cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # # Find the number of principal components that reach 95% explained variance
    # n_components = np.argmax(cumulative_explained_variance >= 0.75) + 1
    # print(f'Number of components for 95% explained variance: {n_components}')

    # plot_pca_explained_variance(cumulative_explained_variance)
    
    # return


    # Save the PCA model
    joblib.dump(pca, './data/%s_pca.pkl'%save_model_prefix_name)   

    # Convert the result to a DataFrame
    principal_df = pd.DataFrame(data = principal_components, columns = [f"PC{i+1}" for i in range(n_components)])

    # Get the principal component weights
    component_weights = pca.components_

    # Create a DataFrame to inspect each feature's contribution to the principal components
    weights_df = pd.DataFrame(component_weights, columns=features_df.columns)

    # Sort the feature contributions for the first principal component (PC1)
    top_features_pc1 = weights_df.iloc[0].abs().sort_values(ascending=False)[:n_components]

    # Get the column names of these features
    top_feature_names = top_features_pc1.index.tolist()

    print("Top contributing features to PC1:", top_feature_names)

    return principal_df,principal_components

if __name__ == '__main__':
    n_components=13
    ib_df = pd.read_csv('./data/ib600_2_features.csv')
    ob_df = pd.read_csv('./data/ob600_2_features.csv')
    tb_df = pd.read_csv('./data/tb600_2_features.csv')
    normal_df = pd.read_csv('./data/n600_3_2_features.csv')

    features_df = pd.concat([ib_df, normal_df])
    principal_df,principal_components = process_data(features_df,n_components=n_components,save_model_prefix_name='ib')
    principal_df['label'] = 0
    principal_df.loc[0:(ib_df.shape[0]-1),'label'] = 1
    principal_df.to_csv('./data/ib600_2_features_pca%d.csv'%n_components,index=False)
    show_pca_results(principal_components,principal_df['label'])

    features_df = pd.concat([tb_df, normal_df])
    principal_df,principal_components = process_data(features_df,n_components=n_components,save_model_prefix_name='tb')
    principal_df['label'] = 0
    principal_df.loc[0:(tb_df.shape[0]-1),'label'] = 1
    principal_df.to_csv('./data/tb600_2_features_pca%d.csv'%n_components,index=False)

    features_df = pd.concat([ob_df, normal_df])
    principal_df,principal_components = process_data(features_df,n_components=n_components,save_model_prefix_name='ob')
    principal_df['label'] = 0
    principal_df.loc[0:(ob_df.shape[0]-1),'label'] = 1
    principal_df.to_csv('./data/ob600_2_features_pca%d.csv'%n_components,index=False)

    features_df = pd.concat([ib_df,tb_df,ob_df, normal_df])
    principal_df,principal_components = process_data(features_df,n_components=n_components,save_model_prefix_name='all')
    principal_df['label'] = 0
    principal_df.loc[0:(ib_df.shape[0]-1),'label'] = 1
    principal_df.loc[(ib_df.shape[0]):((ib_df.shape[0])+tb_df.shape[0]-1),'label'] = 2
    principal_df.loc[((ib_df.shape[0])+tb_df.shape[0]):((ib_df.shape[0])+tb_df.shape[0]+ob_df.shape[0]-1),'label'] = 3
    principal_df.to_csv('./data/all600_2_features_pca%d.csv'%n_components,index=False)
    