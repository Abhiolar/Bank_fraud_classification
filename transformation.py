import pandas as pd
import numpy as np
from clean import cleaned
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer


def preprocess(data,labels):

    """Paramters:
    data: This the transaction data, contains all features needed for pre-processing

    labels: This contains the fraudulent data labels and reported Time

    returns: This function returns a fully pre-processed data ready for model training"""
    new_df = cleaned(data,labels)

    "Seperate continuous and categorical features from the dataframe"


    cat_features = new_df[['transaction_month', 'merchantCountry', 'mcc', 'posEntryMode']]
    cont_features = new_df[['transactionAmount','availableCash']]
    target = new_df['label']


    "apply string datatype to all the categorical features"

    cat_features = cat_features.applymap(str)

    return cat_features,cont_features,target



def normalisation(cat_features,cont_features):
    """This function performs log transformation and feature hashing on the
    continuous and categorical features respectively"""

    """Paramters:
    Cont_faetures: This contains pre-processed continuous cont_features

    cat_features: This contains the pre-processed categorical features

    returns: The normalised dataframe"""


    "Log transformation of continuous variables"

    log_cont = (np.log(cont_features + 1))

    """After normalising the variables,scale the variables using MinMax scaler"""



    scaler = MinMaxScaler()
    scaled_cont = scaler.fit_transform(log_cont)

    "Apply FeatureHasher function on categorical variables"

    orig_features = cat_features.shape[1]
    hash_vector_size = 6
    ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features = hash_vector_size,
                                                input_type ='string'),i) for i in range(orig_features)])

    hasher = ct.fit_transform(cat_features)

    full_df = pd.concat([pd.DataFrame(scaled_cont,columns = ['transaction_amount','available_cash']),
           pd.DataFrame(hasher).reindex(pd.DataFrame(scaled_cont).index)], axis=1)


    return full_df
