#!/usr/bin/env python

"""
Train script
"""

import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train-dataset-path', help='Hyperpartisan news dataset path (Train)')
    parser.add_argument('--val-dataset-path', help='Hyperpartisan news dataset path (Validation)')
    parser.add_argument('--test-dataset-path', help='Hyperpartisan news dataset path (Test)')
    parser.add_argument('name', help='Classifier\'s name')

    args = vars(parser.parse_args())
    return args['train-dataset-path'], args['val_dataset_path'], args['test_dataset_path'], args['name']


def load_dataset(path):
    dataset = np.load(path)
    return dataset['X'], dataset['y']


def assess_performance(clf, X, y, cv=10):
    cv = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=True
    )
    
    for metric, vals in cv.items():
        vals = np.array(vals) * 100
        mean = np.mean(vals)
        variance = np.var(vals)
        print('{:10}\t: mean: {:4.4} ; var: {:4.4}'.format(metric, mean, variance))
        
    return cv


if __name__ == '__main__':
    train_dataset_path, val_dataset_path, test_dataset_path, name = parse_args()

    X, y = load_dataset(train_dataset_path)
    X = X.astype(np.float32)
    y = y.astype(np.float32) #.reshape((-1, 1))

    ## Train Random Forest
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        min_samples_leaf=1,
        min_samples_split=2
    )

    ## Cross-Validate
    from sklearn.model_selection import StratifiedKFold
    cv_split = StratifiedKFold(n_splits=10)
    cv = assess_performance(classifier, X, y, cv=cv_split)

    ## Save classifier
    import pickle, random
    pickle.dump(classifier, open('../models/{}.pickle'.format(name if name is not None else 'classifier' + str(random.randint(0, 100))), 'wb'))
