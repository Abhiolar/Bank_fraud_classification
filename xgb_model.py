import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from confusion_mx import confusion
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report


def xgb_smote_recall(data,target):
    """This function returns the confusion matrix and
    evaluation metrics for the XGB classifier whilst using SMOTE """

    "Instantaite smote object"

    smote = SMOTE()

    "Train, test, split on the features"

    Xsmote_train,Xsmote_test,ysmote_train,ysmote_test = train_test_split(data, target, test_size = 0.3, random_state = 42)

    "Fit the data to the smote object"

    Xsmote_train_resampled, ysmote_train_resampled = smote.fit_sample(Xsmote_train, ysmote_train)


    clf = XGBClassifier()


    param_grid = {
    'learning_rate': [0.1],
    'max_depth': [3,6],
    'min_child_weight': [1,3],
    'subsample': [0.3,0.7],
    'n_estimators': [25],
    }




    "Optimise classifier for recall(optimise against false negatives)"

    grid_clf = GridSearchCV(clf, param_grid, scoring='recall', cv=3, n_jobs=1)
    grid_clf.fit(Xsmote_train_resampled, ysmote_train_resampled)

    best_parameters = grid_clf.best_params_
    print('Grid Search found the following optimal parameters: ')
    for param_name in sorted(best_parameters.keys()):
      print('%s: %r' % (param_name, best_parameters[param_name]))

    training_preds = grid_clf.predict(Xsmote_train_resampled)
    test_preds = grid_clf.predict(np.array(Xsmote_test))

    training_accuracy = accuracy_score(ysmote_train_resampled, training_preds)
    test_accuracy = accuracy_score(ysmote_test, test_preds)

    training_recall = recall_score(ysmote_train_resampled, training_preds)
    test_recall = recall_score(ysmote_test, test_preds)

    training_precision = precision_score(ysmote_train_resampled, training_preds)
    test_precision = precision_score(ysmote_test, test_preds)

    print('')
    print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
    print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))

    print('')
    print('Training recall: {:.4}%'.format(training_recall * 100))
    print('Validation recall: {:.4}%'.format(test_recall * 100))

    print('')
    print('Training precision: {:.4}%'.format(training_precision * 100))
    print('Validation precision: {:.4}%'.format(test_precision * 100))

    """Use confusion matrix function to get the number of
    predicted positive and negative cases"""

    cnf_matrix_7 = confusion_matrix(ysmote_test,test_preds)
    print('Confusion Matrix for Optimal Gradient Boosted Classifier:\n', cnf_matrix_7)

    confusion(cnf_matrix_7, target)
