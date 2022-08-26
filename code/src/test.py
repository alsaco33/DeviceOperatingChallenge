import sys
import pandas as pd
import numpy as np
import time
import pickle
from catboost import CatBoostClassifier
from aux_functions import extract_position_features, extract_window_features

nn_var_groups = [['accel_z_10_min', 'accel_z_10_ewm', 'accel_y_5_ewm', 'accel_y_5_min', 'accel_x_100_t3']]

knn_targets = [[i for i in range(20)],
               [9,10],
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
               [19,5]]

def main():
    """
    Usage: python -W ignore test.py <testing_filename> <solution_filename> <model_filename>
    Example: python -W ignore test.py "../data/raw/testing_provisional.csv" "../data/results/solution.csv" "../models/model_name"
    """

    # 0. Load parameters of the execution
    testing_filename = sys.argv[1]
    solution_filename = sys.argv[2]
    model_filename = sys.argv[3]

    # 1. Read testing data
    print(time.ctime(), 'Loading raw test data')
    testdf = pd.read_csv(testing_filename)
    print(time.ctime(), 'Loaded test data with shape', testdf.shape)
    
    # 2. Load model objects
    print(time.ctime(), 'Loading model objects')
    
    knns = {}
    for i in range(len(nn_var_groups)):
        for j in range(len(knn_targets)):
            knns[f'{i}_{j}'] = pickle.load(open(f'{model_filename}_knn_{i}_{j}_fold_0', 'rb'))
    
    models = {}
    n_models = 5
    for i in range(n_models):
        models[i] = CatBoostClassifier()
        models[i].load_model(f'{model_filename}_{i}')
        
    # 3. Extract features and compute model predictions window by window
    sessions = testdf["session_id"].unique()
    lookback = 300
    predictions = pd.DataFrame()

    for session in sessions:
        print(time.ctime(), 'Started processing session', session)
        # Filter dataframe of the selected session
        currentdf = testdf[testdf['session_id']==session].reset_index(drop=True)
        
        # Compute the accumulated position vector from gyro data
        currentdf = extract_position_features(currentdf)
        
        # Treat each window separately within the filtered session
        current_row = 0
        while current_row < currentdf.shape[0]:
            print(time.ctime(), 'Processing row', current_row)
            start = max(current_row - lookback, 0)
            window_size = min(26, currentdf.shape[0] - current_row)
            
            # Compute features for the current window
            print(time.ctime(), 'Extracting window features')
            currentfeatures = extract_window_features(currentdf[start:current_row+window_size], window_size)
            currentfeatures.replace([np.inf, -np.inf], np.nan, inplace=True)
            currentfeatures = currentfeatures.fillna(0)
                    
            # Include KNN predictions
            print(time.ctime(), 'Including KNN predictions')
            for i in range(len(nn_var_groups)):
                for j in range(len(knn_targets)):
                    if len(knn_targets[j])>2:
                        currentfeatures[f'feat_cat_knn_class_{i}_{j}'] = knns[f'{i}_{j}'].predict(currentfeatures[nn_var_groups[i]])
                    else:
                        currentfeatures[f'feat_knn_class_{i}_{j}'] = knns[f'{i}_{j}'].predict_proba(
                            currentfeatures[nn_var_groups[i]])[:,0]
                        
            # Compute predictions for the current window
            print(time.ctime(), 'Using meta-model')
            currentpredictions = pd.DataFrame(models[0].predict_proba(currentfeatures[models[0].feature_names_]),
                                              columns = [f'class_{i}' for i in range(20)]) / n_models
            for j in range(1, n_models):
                currentpredictions += pd.DataFrame(models[j].predict_proba(currentfeatures[models[j].feature_names_]),
                                                   columns = [f'class_{i}' for i in range(20)]) / n_models
            currentpredictions.insert(0, 'timestep', currentfeatures['timestep'].values)
            currentpredictions.insert(0, 'session_id', currentfeatures['session_id'].values)
            predictions = pd.concat([predictions, currentpredictions])
            current_row += window_size

    # 4. Store the results in solution_filename
    predictions.to_csv(solution_filename, index=False)
    
if __name__ == "__main__":
    main()