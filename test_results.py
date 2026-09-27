import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import joblib
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors

import random

def show_pca_results(X_pca,y):

    # Use t-SNE to reduce 13-dimensional data to 3D
    # tsne = TSNE(n_components=3, random_state=0)
    # X_tsne = tsne.fit_transform(X_pca)

    # Create a new 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D scatter plot
    sc = ax.scatter(X_pca[:, 0],X_pca[:, 2], X_pca[:, 3], c=y,alpha=0.5,edgecolor='k', s=40, cmap='plasma')

    # Set axis labels
    ax.set_xlabel('Absolute energy')
    ax.set_ylabel('Centroid')
    ax.set_zlabel('ECDF Percentile')

    handles, labels = sc.legend_elements()
    new_labels = [f"Feature Group {int(float(l.split('{')[1].split('}')[0])) + 1}" for l in labels]

    # Add legend
    legend1 = ax.legend(*sc.legend_elements(), 
                    loc="upper right", 
                    bbox_to_anchor=(0.95, 0.75), # Adjust these two values (x, y) to move the position
                    title="Sample Groups")
    plt.gca().add_artist(legend1)

    # Display the figure
    plt.show()


def show_results(test_y, pred_y,average=None,model_name='CE-SiDiag'):

    # Accuracy
    accuracy = accuracy_score(test_y, pred_y)
    print('%s Accuracy: %.2f%%' % (model_name, accuracy * 100.0))

    # Calculate F1 score. Default is for binary classification; averaging options: 'micro', 'macro', 'weighted', or 'samples'. 'macro' is used here as an example.
    if average:
        f1 = f1_score(test_y, pred_y,average=average)
    else:
        f1 = f1_score(test_y, pred_y)
    print('%s F1: %.2f%%' % (model_name, f1 * 100.0))

    # Confusion matrix
    conf_mat = confusion_matrix(test_y, pred_y)

    # Convert confusion matrix to DataFrame for easier viewing and plotting
    if average:
        conf_mat_df = pd.DataFrame(conf_mat, 
                            index=['Actual 0', 'Actual 1','Actual 2','Actual 3'], 
                            columns=['Predicted 0', 'Predicted 1','Predicted 2','Predicted 3'])
    else:
        conf_mat_df = pd.DataFrame(conf_mat, 
                                index=['Actual 0', 'Actual 1'], 
                                columns=['Predicted 0', 'Predicted 1'])

    # Print confusion matrix
    print("%s Confusion Matrix:\n"%model_name, conf_mat_df)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_df, annot=True, fmt='d', cmap='Blues')
    plt.title('%s Confusion Matrix'%model_name)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Plot PCA cumulative explained variance chart
def plot_pca_explained_variance(cumulative_explained_variance):

    # Plot cumulative explained variance
    plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    plt.show()

def plot_time_series_single(windows_list, data_type):
    # Create subplot
    plt.figure(figsize=(12, 12))

    # Plot time series
    plt.plot(windows_list.index, windows_list, label=f'{data_type}', color='red', linestyle='--')

    # Set chart title and labels
    plt.title(f'{data_type} Time Series Over Time', fontsize=20)
    plt.xlabel('Index', fontsize=16)
    plt.ylabel('Value', fontsize=16)

    plt.show()

# Plot time series window charts based on the specified number of windows
def plot_time_series(window_list, data_type_list, n_rows=2, n_cols=2):
    # Create subplots
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7 * n_cols, 3 * n_rows))

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Generate color list
    colors_list = ['red', 'orange','#d1c907', 'green', 'm']

    # Plot time series
    for i, (df, ax, color) in enumerate(zip(window_list, axs, colors_list)):
        ax.plot(df.index, df, label=f'Window {i + 1}', color=color, linestyle='--')
        data_type = data_type_list[i]

        # Set chart title and labels
        ax.set_title(f'{data_type} Time Series Over Time', fontsize=20)
        ax.set_xlabel('Index', fontsize=16)
        ax.set_ylabel('Value', fontsize=16)
        ax.legend()

    # Close unused subplots
    for j in range(len(window_list), n_rows * n_cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    # Save chart as JPG format
    plt.savefig(f'./doc/{data_type}_time_series.jpg', format='jpg', dpi=300)
    plt.show()

# Plot feature trend charts
def plot_feature_trend(feature_names,features_df):
    show_number=len(feature_names)

    # Create subplots
    fig, axs = plt.subplots(nrows=show_number, figsize=(14, 3 * show_number))

    colors_list = ['r','#2980b9','g','c','m']

    # Plot time series
    for i in range(len(feature_names)):
        feature_name=feature_names[i]
        color=colors_list[i]
        ax=axs[i]

        # Plot feature trend chart
        ax.plot(features_df.index, features_df[feature_name], label=feature_name, color=color, linestyle='-')

        # Set chart title and labels
        ax.set_title(f'{feature_name} Trend Over Time')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    plt.show()



