# Import Libraries
import pandas as pd
import numpy as np
import pickle
from itertools import product

# Model development
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from random import sample
import xgboost as xgb
from xgboost import plot_importance

# Plot
import matplotlib.pyplot as plt
import seaborn as sns


# Sample splitting
def split_data(data, TARGET_VAR, SEP_VAR, RANDOM_STATE, SAMPLE_RATIO):
    """Split data into train, val and test set where each set contains features (X) and labels (y)
    Input
    data: data set
    TARGET_VAR: target variable name
    SEP_VAR: split sample by which variable. If SEP_VAR = "", then split by row
    RANDOM_STATE: random seed
    SAMPLE_RATIO: a list which is (train ratio, val ratio)
    """
    
    if SEP_VAR == "": # not specify a seperating variable, then sampling by row
        data['_index'] = data.reset_index().index
        SEP_VAR = '_index'
    
    # Random Selection on SEP_VAR
    pat = pd.DataFrame(data[SEP_VAR].unique())
    samplelist = pat.sample(frac=1, random_state=RANDOM_STATE, replace=False)
    samplelist = samplelist[0].values

    # Split train, dev, test
    X_train = data[data[SEP_VAR].isin(
        samplelist[:round(len(samplelist) * SAMPLE_RATIO[0])])].drop(
            columns=[TARGET_VAR, SEP_VAR])
    X_dev = data[data[SEP_VAR].isin(
        samplelist[round(len(samplelist) *
                         SAMPLE_RATIO[0]):round(len(samplelist) * sum(SAMPLE_RATIO))])].drop(
                             columns=[TARGET_VAR, SEP_VAR])
    X_test = data[data[SEP_VAR].isin(
        samplelist[round(len(samplelist) * sum(SAMPLE_RATIO)):])].drop(
            columns=[TARGET_VAR, SEP_VAR])

    y_train = data[data[SEP_VAR].isin(
        samplelist[:round(len(samplelist) * SAMPLE_RATIO[0])])][TARGET_VAR]
    y_dev = data[data[SEP_VAR].isin(
        samplelist[round(len(samplelist) * SAMPLE_RATIO[0]):round(len(samplelist) *
                                                       sum(SAMPLE_RATIO))])][TARGET_VAR]
    y_test = data[data[SEP_VAR].isin(samplelist[round(len(samplelist) *
                                                      sum(SAMPLE_RATIO)):])][TARGET_VAR]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


# Train model
def fit_model(X_train, y_train, X_dev, y_dev, params, others):
    """Train model
    Input
    X_train: features of training data
    y_train: labels of training data
    X_dev: features of dev data, used for early stopping
    y_dev: labels of dev data
    params: model hyperparameters
    others: other parameters of model, e.g. random_state, eval_metric, earlystop
    """
    
    estimator = xgb.XGBRegressor(**params, random_state=others['random_state'],
                                 objectives='reg:squarederror')
    estimator.fit(X_train,
                y_train,
                eval_set=[(X_dev, y_dev)],
                eval_metric=others['eval_metric'],
                early_stopping_rounds=others['earlystop'],
                verbose=False)
    
    return estimator


# Evaluate model
def eval_model(model, X, y, _print=False):
    """Evaluate model performance
    Input
    model: trained model
    X: features 
    y: target or label
    _print: True or False. True is to print evaluation metrics
    """
    
    pred = pd.DataFrame({'y': y, 'pred_y': model.predict(X)})
    
    mae = metrics.mean_absolute_error(pred['y'], pred['pred_y'])
    rmse = np.sqrt(metrics.mean_squared_error(pred['y'], pred['pred_y']))
    r2 = metrics.r2_score(pred['y'], pred['pred_y'])
    
    if _print == True:
        print(f'MAE: {round(mae, 3)}, RMSE: {round(rmse, 3)}, R2: {round(r2, 3)}')
    
    return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'r2': round(r2, 3)}



# Compare model performance with different hyperparameters
def grid_search_result(grid_params, X_train, y_train, X_dev, y_dev, MODEL_OTHERS):
    """Create grid search table of model performance for different hyperparameters
    Input
    grid_params: a dictionary which is composed by different choices of hyperparameters
    X_train: features of training data
    y_train: labels of training data
    X_dev: features of dev data, used for early stopping
    y_dev: labels of dev data
    MODEL_OTHERS: other hyperparameters of model
    """
    
    # All combinations of hyperparameters
    params_combs = list(product(*grid_params.values()))
    
    _create_result_tbl = True  # To create result table in the first loop 
    for params_value in params_combs:
        if _create_result_tbl == True:
            # convert list of hyperparameters into a dictionary 
            params = {key:value for key,value in zip(grid_params.keys(), params_value)}
            
            # Train model: evaluate on both train and dev data
            model = fit_model(X_train, y_train, X_dev, y_dev, params, MODEL_OTHERS)
            perf_train = eval_model(model, X_train, y_train, _print=False)
            perf_val = eval_model(model, X_dev, y_dev, _print=False)

            # Create empty result table
            result = pd.DataFrame(columns=list(grid_params.keys()) + [
                'train_' + i for i in perf_train.keys()] + ['val_' + i for i in perf_train.keys()])
            
            # Add row
            params.update({'train_' + key:value for key, value in perf_train.items()})
            params.update({'val_' + key:value for key, value in perf_val.items()})
            result = result.append(params, ignore_index=True)

            # Change flag value
            _create_result_tbl = False
        else:
            params = {key:value for key,value in zip(grid_params.keys(), params_value)}
            model = fit_model(X_train, y_train, X_dev, y_dev, params, MODEL_OTHERS)
            perf_train = eval_model(model, X_train, y_train, _print=False)
            perf_val = eval_model(model, X_dev, y_dev, _print=False)
            
            params.update({'train_' + key:value for key, value in perf_train.items()})
            params.update({'val_' + key:value for key, value in perf_val.items()})
            result = result.append(params, ignore_index=True)
            
    return result
