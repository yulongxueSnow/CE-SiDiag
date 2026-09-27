import pdb
import joblib
from sklearn.preprocessing import StandardScaler
from naive_bayes_classfier import BayesModel
from chaos_classfier import ChaosEnsembleModel
from similar_vector import *
from test_results import *

def split_data(df):
    X = df.drop('label', axis=1)  # Feature data, excluding the 'label' column
    y = df['label']  # Target variable, i.e., the 'label' column
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    return train_x, test_x, train_y, test_y

def trans_pca_features(feature_df,mode,n_components):
    pca_scaler_model_path = './data/%s_pca_scaler.pkl'%mode
    pca_model_path = './data/%s_pca.pkl'%mode

    # Load the saved normalization parameters
    scaler_loaded = joblib.load(pca_scaler_model_path)

    # Load the saved PCA model
    pca_loaded = joblib.load(pca_model_path)

    scaled_features = scaler_loaded.transform(feature_df)
    principal_components = pca_loaded.fit_transform(scaled_features)

    # Convert the result to a DataFrame
    principal_df = pd.DataFrame(data = principal_components, columns = [f"PC{i+1}" for i in range(n_components)])

    # Release resources
    del scaler_loaded
    del pca_loaded

    return principal_df

if __name__ == '__main__':
    n_components=13
    ib_df = pd.read_csv('./data/ib600_2_features.csv')
    ob_df = pd.read_csv('./data/ob600_2_features.csv')
    tb_df = pd.read_csv('./data/tb600_2_features.csv')
    normal_df = pd.read_csv('./data/n600_3_2_features.csv')
    ib_df['label'] = 1
    ob_df['label'] = 2
    tb_df['label'] = 3
    normal_df['label'] = 0

    chaos_map = {'chaos_map': 'sine','chaos_iterations':10,'r':0.99}
    pred_y = []
    test_y = []

    data_fpaths=['./data/ib600_2_features_pca%d.csv'%n_components,
                './data/tb600_2_features_pca%d.csv'%n_components,
                './data/ob600_2_features_pca%d.csv'%n_components]
    for data_fpath in data_fpaths:
        if 'ib' in data_fpath:
            mode='ib'
        elif 'ob' in data_fpath:
            mode='ob'
        elif 'tb' in data_fpath:
            mode='tb'
        else:
            mode=''
        model_obj = ChaosEnsembleModel(data_fpath)
        chaos_train_x, chaos_test_x, chaos_train_y, chaos_test_y, chaos_y_predict, accuracy, df_columns = model_obj.exc_p(mode,chaos_map)
        print('%s chaos ensemble models accuracy is :%.5f'%(mode,accuracy))

        test_y_list = chaos_test_y.tolist()
        for index in range(len(chaos_y_predict)):
            label = int(test_y_list[index])
            predict_y = int(chaos_y_predict[index])
            most_similar_index, max_similarity = find_most_similar(chaos_test_x[index],chaos_train_x)
            similar_predict = int(chaos_train_y[most_similar_index])
            if max_similarity >= 0.9:
                predict_y = similar_predict

            if predict_y == 1:
                if mode == 'ob':
                    predict_y = 2
                elif mode == 'tb':
                    predict_y = 3
            if label == 1:
                if mode == 'ob':
                    label = 2
                elif mode == 'tb':
                    label = 3
            test_y.append(label)
            pred_y.append(predict_y)

    show_results(test_y, pred_y,average='macro')

    ###########################################################################
    # Similarity vector prediction
    ###########################################################################

    df = pd.concat([ib_df,tb_df,ob_df, normal_df])
    df = df.reset_index(drop=True)

    principal_df = trans_pca_features(df.drop('label', axis=1),'all',n_components)
    principal_df['label'] = 0
    principal_df.loc[0:(ib_df.shape[0]-1),'label'] = 1
    principal_df.loc[(ib_df.shape[0]):((ib_df.shape[0])+tb_df.shape[0]-1),'label'] = 2
    principal_df.loc[((ib_df.shape[0])+tb_df.shape[0]):((ib_df.shape[0])+tb_df.shape[0]+ob_df.shape[0]-1),'label'] = 3

    # Split the test set
    train_x, test_x, train_y, test_y = split_data(principal_df)

    pred_y_list = []
    prob_list = []

    show_example = {
        0:0,
        1:0,
        2:0,
        3:0
    }

    # Draw four subplots, showing one similar-data subplot for each type
    plt.figure(figsize=(12, 4))
    for index, row in test_x.iterrows():
        most_similar_index, max_similarity=find_most_similar(row.values, train_x.values)
        pred_y_list.append(train_y.iloc[most_similar_index])
        prob_list.append(max_similarity)
        label = train_y.iloc[most_similar_index]

        # Display the trend chart of the two similar data points
        if show_example[label]==0:
            
            # Create the subplot
            plt.subplot(1, 4, label + 1)

            # The title shows the most similar fault label
            if label==0:
                plt.title('Similar Data--Normal,Prob %.5f'%max_similarity)
            elif label==1:
                plt.title('Similar Data--IB,Prob %.5f'%max_similarity)
            elif label==2:
                plt.title('Similar Data--OB,Prob %.5f'%max_similarity)
            elif label==3:
                plt.title('Similar Data--TB,Prob %.5f'%max_similarity)
            else:
                continue
            show_example[label]+=1

            plt.plot(range(len(row)), row.values,'r--',label='new data')
            plt.plot(range(len(train_x.iloc[most_similar_index].values)), train_x.iloc[most_similar_index].values,'g--',label='similar')
            plt.legend()
    plt.tight_layout()  
    plt.show()

    pred_y = np.array(pred_y_list)
    show_results(test_y, pred_y,average='macro',model_name='similar vector')

   