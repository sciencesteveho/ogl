#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to train machine-learning regressors on the k-eigenvalues of the graph laplacian as a baseline to compare GNN performance.

Models implemented:
    Random forest
    SVM
    gaussian mixed model
    ordinary linear regression.
"""

import os
import pickle
import numpy as np
from joblib import dump, load

### import baseline models
from sklearn.svm import LinearSVC
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

from sklearn.metrics import mean_squared_error


def _set_matplotlib_publication_parameters():
    plt.rcParams.update({'font.size': 7})  # set font size
    plt.rcParams["font.family"] = 'Helvetica'  # set font


def unpickle(file):
    '''
    Unpickles an object
    Params
        file: str, name of the pickled file
    '''
    with open(file, 'rb') as f:
        var = pickle.load(f)
    return var


def save_pickle(file, obj):
    '''
    Saves an object
    Params
        file: str, name to save file as
        obj: object to save
    '''
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def prepare_vector_dicts(eigen_file, label_file):
    '''
    Prepares dictionaries of the train/val/split validation
    Format
        gene_tissue: (k-eigs vector, regression_target)  # (ndarray, float)
    '''
    eigens = unpickle(eigen_file)
    labels = unpickle(label_file)

    for part in ['train', 'validation', 'test']:
        print(f'starting {part}')
        eigs = {gene: (eigens[gene], labels[part][gene]) for gene in labels[part].keys()}
        save_pickle(f'{part}_eigs_all.pkl', eigs)
        print(f'finishing {part}')


def get_training_data(vector_dict):
    '''
    Get values from dicts into lists for training
    Returns
        list of k-eig vectors, list of target values
    '''
    eigs = unpickle(vector_dict)
    return [tup[0] for tup in eigs.values()], [tup[1] for tup in eigs.values()]  # x_train, y_train


def train_and_predict(regressor, x_train, y_train, x_val, y_val, x_test, y_test, regressor_name):
    '''
    Fits regressors and makes prediction on val and test sets. Prints RMSE value for both tests
    '''
    regressor.fit(x_train, y_train)

    dump(regressor, f'{regressor_name}_nofilt_model.pt')  # save model for plotting

    def pred_rmse(x, y):
        pred = regressor.predict(x)
        return pred, mean_squared_error(y, pred, squared=False)

    val_preds, val_rmse = pred_rmse(x_val, y_val)
    test_preds, test_rmse = pred_rmse(x_test, y_test)

    print(f'{regressor_name} val_rmse: {val_rmse}')
    print(f'{regressor_name} test_rmse: {test_rmse}')
    # return val_preds, val_rmse, test_preds, test_rmse


if __name__ == "__main__":

    ### set vector dictionaries
    # train_file = 'train_eigs.pkl'
    # validation_file = 'validation_eigs.pkl'
    # test_file = 'test_eigs.pkl'
    target_file = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/targets_filtered_2500.pkl'

    train_file = 'train_eigs_all.pkl'
    validation_file = 'validation_eigs_all.pkl'
    test_file = 'test_eigs_all.pkl'

    ### prepare dicts if they do not exist
    if train_file not in os.listdir():
        prepare_vector_dicts(
            eigen_file='/ocean/projects/bio210019p/stevesho/data/preprocess/chunks/eigs/eigs_2500.pkl',
            label_file = target_file,
        )

    ### get training vectors
    x_train, y_train = get_training_data(train_file)
    x_val, y_val = get_training_data(validation_file)
    x_test, y_test = get_training_data(test_file)

    ### set up regressors
    ridge_regressor = linear_model.Ridge(alpha=.5)
    rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    svm_regressor = LinearSVC()
    linear_regressor = LinearRegression()
    gmm_regressor = GaussianMixture(n_components=3, random_state=42, n_init=50)

    ### train models and return prediction RMSE
    train_and_predict(ridge_regressor, x_train, y_train, x_val, y_val, x_test, y_test, 'ridge_regressor')
    train_and_predict(svm_regressor, x_train, y_train, x_val, y_val, x_test, y_test, 'SVM')
    train_and_predict(linear_regressor, x_train, y_train, x_val, y_val, x_test, y_test, 'linear_regressor')
    train_and_predict(rf_regressor, x_train, y_train, x_val, y_val, x_test, y_test, 'random_forest')
    train_and_predict(gmm_regressor, x_train, y_train, x_val, y_val, x_test, y_test, 'gausian_mixed_model')






    ### set vector dictionaries
    train_file = 'train_eigs_all.pkl'
    validation_file = 'validation_eigs_all.pkl'
    test_file = 'test_eigs_all.pkl'

    ### get training vectors
    # x_train, y_train = get_training_data(train_file)
    x_val, y_val = get_training_data(validation_file)
    x_test, y_test = get_training_data(test_file)

    rf_regressor = load('random_forest_nofilt_model.pt')
    ridge_regressor = load('ridge_regressor_nofilt_model.pt')
    linear_regressor = load('linear_regressor_nofilt_model.pt')

    def pred_rmse(x, y, regressor):
        pred = regressor.predict(x)
        return pred, mean_squared_error(y, pred, squared=False)

    rf_test_preds, rf_rmse = pred_rmse(x_test, y_test, rf_regressor)
    ridge_test_preds, ridge_rmse = pred_rmse(x_test, y_test, ridge_regressor)
    linear_test_preds, linear_rmse = pred_rmse(x_test, y_test, linear_regressor)

    save_pickle('rf_test_preds.pkl', rf_test_preds)
    save_pickle('ridge_test_preds.pkl', ridge_test_preds)
    save_pickle('linear_test_preds.pkl', linear_test_preds)

    print(f'rf rmse {rf_rmse}')
    print(f'ridge rmse {ridge_rmse}')
    print(f'linear rmse {linear_rmse}')


    import seaborn as sns
    import matplotlib.pyplot as plt
    linear = unpickle('linear_test_preds.pkl')
    rf = unpickle('rf_test_preds.pkl')
    ridge = unpickle('ridge_test_preds.pkl')

    _set_matplotlib_publication_parameters()


    linear = linear.flatten()
    rf = rf.flatten()
    ridge = ridge.flatten()

    y_test = np.vstack(y_test).flatten()

    ax = sns.regplot(y_test, linear, ci=None, marker='.')
    plt.xlabel('Expected')
    plt.ylabel('Prediction')
    plt.title('Linear regressor')
    plt.show()

    ax = sns.regplot(y_test, rf, ci=None, marker='.')
    plt.xlabel('Expected')
    plt.ylabel('Prediction')
    plt.title('Random forest')
    plt.show()

    ax = sns.regplot(y_test, ridge, ci=None, marker='.')
    plt.xlabel('Expected')
    plt.ylabel('Prediction')
    plt.title('Ridge regressor')
    plt.show()