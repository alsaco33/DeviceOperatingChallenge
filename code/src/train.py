import sys
import pandas as pd
import numpy as np
import time
import os
import pickle
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from aux_functions import *

# Define global training variables
nn_var_groups = [['accel_z_10_min', 'accel_z_10_ewm', 'accel_y_5_ewm', 'accel_y_5_min', 'accel_x_100_t3']]

knn_targets = [[i for i in range(20)],
               [9,10],
               [5,4],
               [2,5,4],
               [11,2,4,6],
               [11,4],
               [11,8],
               [11,2],
               [11,6],
               [12,7],
               [12,3],
               [12,14],
               [15,5],
               [15,10],
               [17,9],
               [17,10],
               [18,1],
               [18,13],
               [18,13, 1],
               [19,5],
               [19,5,15]]

def main():
    """
    
    Usage: python -W ignore train.py <training_filename> <model_filename>
    Example: python -W ignore train.py "../data/training.csv" "../model/model_name"
    """
    
    # 0. Load the parameters of the execution
    training_filename = sys.argv[1]
    model_filename = sys.argv[2]

    # 1. Read training data
    print(time.ctime(), 'Reading raw training data')
    traindf = pd.read_csv(training_filename)
    
    # 2. Extract training features window by window in parallel by session
    print(time.ctime(), 'Creating the master table')
    sessions = traindf['session_id'].unique()
    n_sessions = traindf['session_id'].nunique()
    tables = Parallel(n_jobs=7)(delayed(extract_all_features)(traindf[traindf['session_id']==session]) for 
                                session in sessions)
    masterTable = pd.concat(tables)
    masterTable.replace([np.inf, -np.inf], np.nan, inplace=True)
    masterTable = masterTable.fillna(0).reset_index(drop=True)
    print(time.ctime(), 'Created master table with shape', masterTable.shape)
    
    # 3. Perform train-val-test split
    print(time.ctime(), 'Selecting CV fold split')
    np.random.seed(42)
    fold_map = dict(zip(sorted(sessions), 
                        np.random.randint(low=0, high=5, size=n_sessions)))
    
    # 4. Extract KNN features
    print(time.ctime(), 'Extracting KNN features')
    for i in range(len(nn_var_groups)):
        for j in range(len(knn_targets)):
            for k in set(fold_map.values()):
                print(time.ctime(), i, j, k)
                
                train_sessions = [s for s in fold_map.keys() if fold_map[s]!=k]
                
                # Define KNN classifier
                knn_multiclass = KNeighborsClassifier(n_neighbors=200)
                
                # Fit KNN
                knn_multiclass.fit(masterTable.loc[(masterTable['session_id'].isin(train_sessions)) &
                                                   (masterTable['region_class'].isin(knn_targets[j])), nn_var_groups[i]],
                                   masterTable[(masterTable['session_id'].isin(train_sessions)) &
                                               (masterTable['region_class'].isin(knn_targets[j]))]['region_class'])
                
                # Save the cross validation predictions
                if len(knn_targets[j])>2:
                    masterTable.loc[masterTable['session_id'].isin(train_sessions)==False, 
                                    f'feat_cat_knn_class_{i}_{j}'
                                   ] = knn_multiclass.predict(masterTable.loc[
                        masterTable['session_id'].isin(train_sessions)==False, nn_var_groups[i]])
                else:
                    masterTable.loc[masterTable['session_id'].isin(train_sessions)==False, 
                                    f'feat_knn_class_{i}_{j}'] = knn_multiclass.predict_proba(masterTable.loc[
                        masterTable['session_id'].isin(train_sessions)==False, nn_var_groups[i]])[:,0]
                
                # Save the KNN model of fold 1 to use in testing
                if k==0:
                    knnPickle = open(f'{model_filename}_knn_{i}_{j}_fold_{k}', 'wb') 
                    pickle.dump(knn_multiclass, knnPickle)  
                    knnPickle.close()
                
    for col in masterTable.columns:
        if '_cat_' in col:
            masterTable[col] = masterTable[col].astype(int)
    
    # 5. Train and save the catboost multiclass meta-model for every fold
    print(time.ctime(), 'Training the catboost meta-model')
    for i in set(fold_map.values()):
        train_sessions = [k for k in fold_map.keys() if fold_map[k]!=i]
        masterTable['train'] = 0
        masterTable.loc[masterTable['session_id'].isin(train_sessions), 'train'] = 1
        model = train_catboost(masterTable[['session_id', 'timestep', 'region_class', 'train']+SELECTED_VARS], i, model_filename)
        print(time.ctime(), 'Finished training the model')

if __name__ == "__main__":
    main()