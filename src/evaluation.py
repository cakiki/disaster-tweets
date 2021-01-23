# -*- coding: utf-8 -*-
import copy
import dill
import logging as log
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

# Print logging output from debug level and above 
log.basicConfig(level=log.DEBUG)

# The training data file
train_data_file = '../data/external/kaggle/train.csv'
# The test data file for submissions
test_data_file = '../data/external/kaggle/test.csv'
# The random seed for reproducable results in the train-test splits
seed = 42

def evaluate(model, store_model=True, store_submission=True, embeddings=None, preprocessing_func=None, n_folds=10, n_runs=1):
    """Evaluates the given classifier model on the Kaggle competition data set and optionally generates a submission file.
   
    The model must implements the scikit-learn API (i.e. `fit` and `predict`). Walk through the notebook
    `/notebooks/template_model_tutorial.ipynb` for a complete tutorial on how to roll your classifier in this
    challenge and run this evaluation. This method will print training and test performance results
    (F1-Score, accuracy, recall and precision) for the given model obtained through cross-validation (CV).
    Evaluation parameters are intentionally not configurable via parameters in order to keep results across
    different models comparable. Hence, they should only be changed for good reason!

    :param store_model: Whether to store the trained model in `/models`, defaults to `True`
    :type store_model: bool, optional
    :param store_submission: Whether to store a submission file from the model in `/models`, defaults to `True`
    :type store_submission: bool, optional
    :param preprocessing_func: Applies a preprocessing function to the individual entries of data matrix read from the training file. The function is given a row in the data matrix and expects some result that can be used by the given model. Defaults to `None`
    :type preprocessing_func: func, optional
    :param n_folds: The number of folds to perform (change only for debugging), defaults to `10`
    :type n_folds: int, optional
    :param n_runs: The number of experiments to run (change only for debugging), defaults to `1`
    :type n_runs: int, optional
    """
    log.info('Loading training data from {}...'.format(train_data_file))
    # Load data and perform some sanity checks
    df = pd.read_csv(train_data_file)
    if not embeddings:
        X = df[['keyword', 'location', 'text']].values
    else:
        X = np.load(f'../data/features/train_{embeddings}_embeddings.npy')
    y = df['target'].values
    # assert X.shape == (len(y), 3)
    # assert set(np.unique(y)) == set([0, 1])
    log.info('-> Number of samples: {}'.format(len(y)))
    log.info('-> Number of features: {}'.format(X.shape[1]))
    # Apply preprocessing function if provided
    if preprocessing_func is not None:
        log.info('Applying pre-processing function...')
        X = np.array([preprocessing_func(x) for x in X])
        log.info('-> Feature matrix after preprocessing: {}'.format(X.shape))

    # Setup cross-validation (CV) engine
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_runs, random_state=seed)
    log.info('Evaluating model with {} experiment(s) of {}-fold Cross Validation...'.format(n_runs, n_folds))
    # We want to get model results for the ground truth (y_true) and the corresponding model prediction (y_pred)
    train_res = {
        'idx': [],
        'y_true': [],
        'y_pred': []
    }
    test_res = {
        'idx': [],
        'y_true': [],
        'y_pred': []
    }
    # Run CV evaluation and gather up training and test results.
    for r, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        # Get train portion of the data
        X_train, y_train = X[train_idx], y[train_idx]
        # Train on a deep copy of the model to ensure we do not contaminate the evaluation
        cv_model = copy.deepcopy(model)
        cv_model.fit(X_train, y_train)
        # Gather training predictions with corresponding ground truth
        train_res['idx'].extend(train_idx)
        train_res['y_true'].extend(y_train)
        train_res['y_pred'].extend(cv_model.predict(X_train))
        # Gather test predictions with corresponding ground truth
        test_res['idx'].extend(test_idx)
        test_res['y_true'].extend(y[test_idx])
        test_res['y_pred'].extend(cv_model.predict(X[test_idx]))
        # Finished iteration
        log.info('Run {}/{} finished'.format(r+1, n_runs*n_folds))
    
    # We got all results, print out summary
    log.info('---')
    f1_test = metrics.f1_score(test_res['y_true'], test_res['y_pred'])
    log.info('Expected submission results (F1-Score): around {:.2f}'.format(f1_test))
    # Show F1-Score
    f1_train = metrics.f1_score(train_res['y_true'], train_res['y_pred'])
    log.info('F1-Score: {:.2f} (training); {:.2f} (test)'.format(f1_train, f1_test))
    # Show accuracy
    acc_train = metrics.accuracy_score(train_res['y_true'], train_res['y_pred'])
    acc_test = metrics.accuracy_score(test_res['y_true'], test_res['y_pred'])
    log.info('Accuracy: {:.2f}% (training); {:.2f}% (test)'.format(acc_train*100, acc_test*100))
    # Show recall
    recall_train = metrics.recall_score(train_res['y_true'], train_res['y_pred'])
    recall_test = metrics.recall_score(test_res['y_true'], test_res['y_pred'])
    log.info('Recall: {:.2f}% (training); {:.2f}% (test)'.format(recall_train*100, recall_test*100))
    # Show precision
    prec_train = metrics.precision_score(train_res['y_true'], train_res['y_pred'])
    prec_test = metrics.precision_score(test_res['y_true'], test_res['y_pred'])
    log.info('Precision: {:.2f}% (training); {:.2f}% (test)'.format(prec_train*100, prec_test*100))
    
    # Retrain the model on the complete data set to 
    log.info('---')
    log.info('Retraining model on the complete data set...')
    model.fit(X, y)

    # Store the model and/or create a submission file
    if(store_submission or store_model):
        # Retrain model on complete data set
        f1_score = metrics.f1_score(y, model.predict(X))
        log.info('-> F1-Score on complete training set: {:.2f}'.format(f1_score))
        # Create label string for this model
        dt_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        label = '{}_{}_{}x{}cv_{}'.format(dt_str, type(model).__name__, n_runs, n_folds, round(f1_test, 2))
        # Store model with its CV results if flag is set
        if(store_model):
            file_path = f'../models/model_{label}_{embeddings}.pck' if embeddings else f'../models/model_{label}.pck'
            with open(file_path, 'wb') as f:
                dill.dump({'model': model, 'train_res': train_res, 'test_res': test_res}, f)
            log.info('-> Stored model to {}'.format(file_path))
        # Store model with its CV results if flag is set
        if(store_submission):
            # Load test data for submission and compute model predictions
            df_test = pd.read_csv(test_data_file)
            if not embeddings:
                X_subm = df_test[['keyword', 'location', 'text']].values
            else:
                X_subm = np.load(f'../data/features/test_{embeddings}_embeddings.npy')
            if preprocessing_func is not None:
                X_subm = np.array([preprocessing_func(x) for x in X_subm])
            y_subm = model.predict(X_subm)
            # Compile predictions into a submission data frame
            df_subm = pd.DataFrame()
            df_subm['id'] = df_test['id']
            df_subm['target'] = y_subm
            # Save frame as CSV
            file_path = f'../models/submission_{label}_{embeddings}.csv' if embeddings else f'../models/submission_{label}.csv'
            df_subm.to_csv(file_path, index=False)
            log.info('-> Stored submission file to {}'.format(file_path))
    log.info('Evaluation finished.')
    return X, train_res, test_res
