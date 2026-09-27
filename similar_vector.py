import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from test_results import *

def get_most_similar_vectors(vector, vectors_group, train_y, max_counts=100):
    """
    Find the most similar vectors to the given vector and return them.
    
    Parameters:
    - vector: A single vector, type np.ndarray.
    - vectors_group: A group of vectors, type np.ndarray, where each row is a vector.
    
    Returns:
    - The original vector and the most similar vectors.
    """
    # Ensure inputs are NumPy arrays
    vector = np.array(vector)
    vectors_group = np.array(vectors_group)
    
    # Compute cosine similarities
    similarities = np.dot(vectors_group, vector) / (np.linalg.norm(vectors_group, axis=1) * np.linalg.norm(vector))

    # Find the top max_counts most similar vectors and their similarity scores
    sorted_similarities = np.sort(similarities)[-max_counts:]
    most_similar_indices = np.argsort(similarities)[-max_counts:]

    # Get the y labels at these indices
    most_similar_labels = train_y[most_similar_indices]

    # Get all unique labels
    labels = np.unique(train_y)

    # Iterate over all categories to generate average similarity scores per category
    # Format: [[label, level, score]]
    datas = []
    
    # Iterate over sorted_similarities
    for i, similarity in enumerate(sorted_similarities):
        label = most_similar_labels[i]
        score = int(similarity*100)

        # Convert score range 0-1 into 10 similarity levels: >=0.98 is 10, >=0.95 is 9, and so on
        if score >= 98:
            level = 10
        elif score >= 95:
            level = 9
        elif score >= 92:
            level = 8
        elif score >= 90:
            level = 7
        elif score >= 85:
            level = 6
        elif score >= 80:
            level = 5
        elif score >= 75:
            level = 4
        elif score >= 70:
            level = 3
        elif score >= 65:
            level = 2
        elif score >= 60:
            level = 1
        else:
            level = 0

        datas.append([label, level, score])
    

    print("Similarity scores: %s" % datas)

    return similarities, most_similar_indices

def find_most_similar(vector, vectors_group):
    """
    Find the most similar vector to the given vector.
    
    Parameters:
    - vector: A single vector, type np.ndarray.
    - vectors_group: A group of vectors, type np.ndarray, where each row is a vector.
    
    Returns:
    - The index of the most similar vector and its cosine similarity.
    """
    # Ensure inputs are NumPy arrays
    vector = np.array(vector)
    vectors_group = np.array(vectors_group)
    
    # Compute cosine similarities
    similarities = np.dot(vectors_group, vector) / (np.linalg.norm(vectors_group, axis=1) * np.linalg.norm(vector))
    
    # Find the index of the maximum similarity
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[most_similar_index]
    
    return most_similar_index, max_similarity


def batch_cosine_similarity(vectors_group, query_vectors):
    # Normalize both sets of vectors
    vectors_group_normalized = vectors_group / np.linalg.norm(vectors_group, axis=1)[:, np.newaxis]
    query_vectors_normalized = query_vectors / np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]

    # Compute the dot product to get cosine similarities
    similarities = np.dot(query_vectors_normalized, vectors_group_normalized.T)

    return similarities


if __name__ == '__main__':
    n_components=13
    data_fpaths=['./data/all600_2_features_pca%d.csv'%n_components]

    for data_fpath in data_fpaths:

        df = pd.read_csv(data_fpath)
        X = df.drop('label', axis=1)  # Feature data, excluding the 'label' column
        y = df['label']  # Target variable, i.e., the 'label' column
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        pred_y_list = []
        for index, row in test_x.iterrows():
            similarities, most_similar_indices = get_most_similar_vectors(row.values, train_x.values, train_y.values)
            most_similar_index, max_similarity=find_most_similar(row.values, train_x.values)
            pred_y_list.append(train_y.iloc[most_similar_index])
        pred_y = np.array(pred_y_list)
        show_results(test_y, pred_y,average='macro')


    
