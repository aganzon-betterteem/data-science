#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[2]:


def get_model_performance(x_test, y_test, clf):
    """
    Description: generate the performance metric scores of the model
    ---
    Parameters: 
    x_test: DataFrame
    y_test: DataFrame
    clf: model object
    ---
    returns:
    y_pred: DataFrame
    clf_accuracy: str
    clf_confusion_matrix: list
    df_classification_report: DataFrame
    """
    
    # get prediction: y_pred
    y_pred = clf.predict(x_test)
    # get accuracy
    clf_accuracy = metrics.accuracy_score(y_test,y_pred)
    # get confusion matrix
    clf_confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    # get classification report
    df_classification_report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True)).transpose()
    
    return y_pred, clf_accuracy, clf_confusion_matrix, df_classification_report


# In[3]:


def get_model_feature_importance(x_train, clf):
    """
    Description: generate the feature importance based on model training
    ---
    Parameters: 
    clf: classifier model object
    x_train: DataFrame
    ---
    returns df_feature_importance: DataFrame
    """
    f_importance = clf.feature_importances_

    # put into a DataFrame along with feature names for easier understanding.
    f_list = x_train.columns
    df_feature_importance = pd.DataFrame(f_importance, index=f_list, columns=["Importance"])
    return df_feature_importance.sort_values(by='Importance',ascending=False).reset_index().rename(columns={'index':'Factors'})

