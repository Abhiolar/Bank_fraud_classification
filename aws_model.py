"""Note----This script can only be run on an
AWS console"""
"""The aim of this script is to check how well the AWS built in
LinearLearner algorithm fits into our business case"""

"Create notebook instance on AWS console"

"""Import the necsessary amazon sagemaker libraries"""

import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boto3
import sagemaker
from sagemaker import get_execution_role

%matplotlib inline



"""Instantiate sagemaker session object and role IAM
set s3 bucket name to store data and model artifacts"""

# read in the csv file
local_data = 'fraud.csv'

"Pass csv file into a dataframe"
transaction_df = pd.read_csv(local_data)

"Print the shape of the data"
print('Data shape (rows, cols): ', transaction_df.shape)


"This function calculates the fraction of the data that is fraudulent"

def fraudulent_percentage(transaction_df):
    '''Calculate the fraction of all data points that have a 'Class' label of 1; fraudulent.
       :param transaction_df: Dataframe of all transaction data points; has a column 'Class'
       :return: A fractional percentage of fraudulent data points/all points
    '''
    # counts for all classes
    counts = transaction_df['label'].value_counts()

    # get fraudulent and valid cnts
    fraud_cnts = counts[1]
    valid_cnts = counts[0]

    # calculate percentage of fraudulent data
    fraud_percentage = fraud_cnts/(fraud_cnts+valid_cnts)

    return fraud_percentage


"Call the function to calculate the fraud percentage"

fraud_percentage = fraudulent_percentage(transaction_df)

print('Fraudulent percentage = ', fraud_percentage)
print('Total # of fraudulent pts: ', fraud_percentage*transaction_df.shape[0])
print('Out of (total) pts: ', transaction_df.shape[0])


"This function performs the train/test split on the dataset"

def train_test_split(transaction_df, train_frac= 0.7, seed=1):
    '''Shuffle the data and randomly split into train and test sets;
       separate the class labels (the column in transaction_df) from the features.
       :param df: Dataframe of all credit card transaction data
       :param train_frac: The decimal fraction of data that should be training data
       :param seed: Random seed for shuffling and reproducibility, default = 1
       :return: Two tuples (in order): (train_features, train_labels), (test_features, test_labels)
       '''

    # convert the df into a matrix for ease of splitting
    df_matrix = transaction_df.as_matrix()

    # shuffle the data
    np.random.seed(seed)
    np.random.shuffle(df_matrix)

    # split the data
    train_size = int(df_matrix.shape[0] * train_frac)
    # features are all but last column
    train_features  = df_matrix[:train_size, :-1]
    # class labels *are* last column
    train_labels = df_matrix[:train_size, -1]
    # test data
    test_features = df_matrix[train_size:, :-1]
    test_labels = df_matrix[train_size:, -1]

    return (train_features, train_labels), (test_features, test_labels)



"Get the train/test data and pass the values into the variables"

(train_features, train_labels), (test_features, test_labels) = train_test_split(transaction_df, train_frac=0.7)



"""Test that we have correctly split the data and check if train
and test labels are 0s and 1s"""

print('Training data pts: ', len(train_features))
print('Test data pts: ', len(test_features))
print()

# take a look at first item and see that it aligns with first row of data
print('First item: \n', train_features[0])
print('Label: ', train_labels[0])
print()

# test split
assert len(train_features) > 2.333*len(test_features), \
        'Unexpected number of train/test points for a train_frac=0.7'
# test labels
assert np.all(train_labels)== 0 or np.all(train_labels)== 1, \
        'Train labels should be 0s or 1s.'
assert np.all(test_labels)== 0 or np.all(test_labels)== 1, \
        'Test labels should be 0s or 1s.'
print('Tests passed!')


"Build and train the AWS LinearLearner model"

from sagemaker import LinearLearner #import the LinearLearner model

#specify an output path for your model artifacts

prefix = 'fraud'
output_path = 's3://{}/{}'.format(bucket, prefix)

"""Instantiate LinearLearner model(set predictor type to binary classifier),
set positive example weight to balanced, AWS version of smote and set training target
recall to 90%(0.9)"""

linear_balanced = LinearLearner(role=role,
                                train_instance_count=1,
                                train_instance_type='ml.c4.xlarge',
                                predictor_type='binary_classifier',
                                output_path=output_path,
                                sagemaker_session=sagemaker_session,
                                epochs=15,
                                binary_classifier_model_selection_criteria='precision_at_target_recall', # target recall
                                target_recall=0.9, #tune the model for best recall value
                                positive_example_weight_mult='balanced')

"Convert train data to AWS algorithm data format -- record_set"
# convert features/labels to numpy
train_x_np = train_features.astype('float32')
train_y_np = train_labels.astype('float32')

# create RecordSet
formatted_train_data = linear.record_set(train_x_np, labels=train_y_np)


"train the estimator on formatted training data"
%%time
linear_balanced.fit(formatted_train_data)


"Deploy the model to AWS endpoint and create a predictor for the test data"
%%time
balanced_predictor = linear_balanced.deploy(initial_instance_count=1, instance_type='ml.t2.medium')


"""This function evaluates the endpoint on test data
and returns a variety of model metrics"""

def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """

    # We have a lot of test data, so we'll split it into batches of 100
    # split the test data set into batches and evaluate using prediction endpoint
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(test_features, 100)]

    # LinearLearner produces a `predicted_label` for each data point in a batch
    # get the 'predicted_label' for every point in a batch
    test_preds = np.concatenate([np.array([x.label['predicted_label'].float32_tensor.values[0] for x in batch])
                                 for batch in prediction_batches])

    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}




"Get the evaluation metrics for the balanced predictor"
print('Metrics for balanced, LinearLearner.\n')
metrics = evaluate(balanced_predictor,
                   test_features.astype('float32'),
                   test_labels,
                   verbose=True)

"The Result of this test data will be in the slides"

"Always delete predictor endpoint to avoid charges when not in usage"

delete_endpoint(balanced_predictor)
