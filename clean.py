import pandas as pd
import numpy as np



def cleaned(data,labels):

  """"Paramters:

  data: This is the transaction data for the whole dataset

  label: This is the data for just the fraudulent cases

  return: This function returns a clean_data for data modelling purposes"""

  labels['label'] = 1

  "Merge both dataframes"

  new_df = pd.merge(data,labels, on = ['eventId'], how = 'left')

  "fill non-fraudulent transactions with zero label"

  new_df['label'] = new_df['label'].fillna(0.0)


  "Drop the reported Time column as it contains lots of null values"

  new_df.drop(columns = ['reportedTime'], inplace = True)

  "Convert label values to datatype integer"

  new_df['label'] = new_df['label'].astype(int)

  "Extract month from transaction time"

  new_df['transaction_month'] = pd.to_datetime(new_df['transactionTime']).dt.month


  return new_df
