import pandas as pd
import numpy as np
from clean import cleaned
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc,precision_recall_curve
from confusion_mx import confusion
from imblearn.over_sampling import SMOTE


def baseline_regression(data,target):
    """This function returns the evaluation metrics for the
    baseline LogisticRegression model"""


    X_train,X_test,y_train,y_test = train_test_split(data, target, test_size = 0.3, random_state = 42)

    "Fitting a logistic regression model to the train data"

    logreg = LogisticRegressionCV(fit_intercept=False, Cs= 10, solver='liblinear',cv=5)
    model_log = logreg.fit(X_train, y_train)
    model_log

    y_hat_test = logreg.predict(X_test)
    y_hat_train = logreg.predict(X_train)

    residuals = np.abs(y_test - y_hat_test)
    print("test negative and positive examples:",pd.Series(residuals).value_counts())
    print("\n")
    print("normalised values for the predictions:",pd.Series(residuals).value_counts(normalize=True))

    print("\n")

    cnf_matrix = confusion_matrix(y_test,y_hat_test)


    print('Confusion Matrix:\n', cnf_matrix)

    confusion(cnf_matrix,target)

    "Print Evaluation metrics for baseline regression model"
    print('Training Precision: ', precision_score(y_train, y_hat_train))
    print('Testing Precision: ', precision_score(y_test, y_hat_test))
    print('\n')

    print('Training Recall: ', recall_score(y_train, y_hat_train))
    print('Testing Recall: ', recall_score(y_test, y_hat_test))
    print('\n')

    print('Training Accuracy: ', accuracy_score(y_train, y_hat_train))
    print('Testing Accuracy: ', accuracy_score(y_test, y_hat_test))
    print('\n')

    print('Training F1-Score: ', f1_score(y_train, y_hat_train))
    print('Testing F1-Score: ', f1_score(y_test, y_hat_test))


def smote_logistic(data,target):
    """This function returns the AUC score and AUC-ROC curve for synthetic minority
    over_sampling combined LogisticRegression model"""


    "Train, test split on the predictor and target variables"

    X_train,X_test,y_train,y_test = train_test_split(data, target, test_size = 0.3, random_state = 42)

    print('Original class distribution: \n')
    print(target.value_counts())
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train)

    "Preview synthetic sample class distribution"

    print('-----------------------------------------')

    print('Synthetic sample class distribution: \n')

    print(pd.Series(y_train_resampled).value_counts())

    C_param_range = [0.005, 0.1, 0.2, 0.5, 0.8, 1, 1.25, 1.5, 2]
    names = [0.005, 0.1, 0.2, 0.5, 0.8, 1, 1.25, 1.5, 2]
    colors = sns.color_palette('Set2', n_colors=len(names))

    plt.figure(figsize=(10, 8))

    for n, c in enumerate(C_param_range):

        "Fit a model"

        logreg = LogisticRegression(fit_intercept=False, C=c, solver='liblinear')

        model_log = logreg.fit(X_train_resampled, y_train_resampled)

        print(model_log) # Preview model params

        # Predict

        y_hat_test = logreg.predict(X_test)

        y_score = logreg.fit(X_train_resampled, y_train_resampled).decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_score)

        print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))

        print('-------------------------------------------------------')

        lw = 2
        plt.plot(fpr, tpr, color=colors[n],

        lw=lw, label='ROC curve Regularization Weight: {}'.format(names[n]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
