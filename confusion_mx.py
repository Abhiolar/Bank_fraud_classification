import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


""" This function calculates the confusion
    matrix of a classifier model"""
def confusion(cnf_matrix,target):
  plt.imshow(cnf_matrix,  cmap=plt.cm.Blues)

# Add title and axis labels
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

# Add appropriate axis scales
  class_names = set(target) # Get class labels to add to matrix
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

# Add labels to each cell
  thresh = cnf_matrix.max() / 2. # Used for text coloring below
# Here we iterate through the confusion matrix and append labels to our visualization
  for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, cnf_matrix[i, j],
             horizontalalignment='center',
             color='white' if cnf_matrix[i, j] > thresh else 'black')

# Add a legend
  plt.colorbar()
  plt.show()
