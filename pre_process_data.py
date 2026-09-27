
import pandas as pd
import numpy as np
import tsfel
from  tsfel import time_series_features_extractor
import matplotlib.pyplot as plt
import pdb
import datetime
from test_results import plot_feature_trend,plot_time_series,plot_time_series_single

def process_data(df,window_size=600,step=300,data_type=''):

    ##########################
    # Read the time series data
    ##########################
    
    time_series = df.iloc[:, 0].values

    ##########################
    # Implement the sliding window logic
    ##########################

    # Store the features extracted from each window
    features_list = []  

    # Store the data of each window, for later visualization
    windows_list = []  

    for i in range(0, len(time_series) - window_size + 1, step):
        window = time_series[i:i+window_size]
        df_window = pd.DataFrame(window, columns=['Value'])
        windows_list.append(df_window)
        
        # Configure feature extraction for the current window
        config = tsfel.get_features_by_domain()  # Or use a custom configuration
        
        # Extract the features of the current window
        window_features = time_series_features_extractor(config,window)
        
        # Add the features to the list
        features_list.append(window_features)

    # Merge the feature data
    features_df = pd.concat(features_list, ignore_index=True)

    return features_df,windows_list

if __name__ == '__main__':
    ib_df = pd.read_csv('./data/江南大学轴承数据(一)/数据/ib600_2.csv')
    ob_df = pd.read_csv('./data/江南大学轴承数据(一)/数据/ob600_2.csv')
    tb_df = pd.read_csv('./data/江南大学轴承数据(一)/数据/tb600_2.csv')
    normal_df = pd.read_csv('./data/江南大学轴承数据(一)/数据/n600_3_2.csv')

    # Window size
    window_size = 2000  

    # Step size for the sliding window
    step = 600      

    # windows_list
    windows_list = []   

    start_time=datetime.datetime.now()
    ib_features_df,ib_windows_list = process_data(ib_df,window_size,step,'ib')
    ib_features_df.to_csv('./data/ib600_2_features.csv', index=False)
    ob_features_df,ob_windows_list = process_data(ob_df,window_size,step,'ob')
    ob_features_df.to_csv('./data/ob600_2_features.csv', index=False)
    tb_features_df,tb_windows_list = process_data(tb_df,window_size,step,'tb')
    tb_features_df.to_csv('./data/tb600_2_features.csv', index=False)
    normal_features_df,normal_windows_list = process_data(normal_df,window_size,step,'normal')
    normal_features_df.to_csv('./data/n600_3_2_features.csv', index=False)
    end_time=datetime.datetime.now()
    print("Conversion time:",(end_time-start_time).seconds)

    # Display the visualization results
    # Display the window trend chart of the original time series
    windows_list = [
        ib_windows_list[0],
        ob_windows_list[0],
        tb_windows_list[0],
        normal_windows_list[0]
    ]
    plot_time_series(windows_list,['ib','ob','tb','ns'])

    # plot_time_series_single(ib_windows_list[0],'ib')

