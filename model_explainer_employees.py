#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shap
import xgboost


# In[2]:


def model_explainer_training(x_train, y_train):
    """
    Description: training the model explainer using xgboost regressor and SHAP
    ---
    Parameters: 
    x_train: DataFrame
    y_train: DataFrame
    ---
    returns explainer: SHAP explainer object
    """
    # train an XGBoost model
    model_explainer = xgboost.XGBRegressor().fit(x_train, y_train)

    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model_explainer)

    return explainer


# In[3]:


def model_explanation_initialization(x, explainer):
    """
    Description: get the feature explanation from x features
    ---
    Parameters: 
    x: DataFrame
    explainer: SHAP explainer object
    ---
    returns df_init_explainer: DataFrame
    """
    # get the explainer based on x features
    shap_values = explainer(x)
    df_init_explainer = pd.DataFrame(shap_values.values, columns=x.columns)
    
    return df_init_explainer


# In[4]:


def model_explainer_transpose(df_init_explainer):
    """
    Description: transpose the values of model explainer into digestible factor-score columns
    ---
    Parameters: 
    df_init_explainer: DataFrame
    ---
    returns df_final_explainer: DataFrame
    """
    # initialization: factors
    list_factor_1 = []
    list_factor_2 = []
    list_factor_3 = []
    list_factor_4 = []
    list_factor_5 = []
    list_factor_6 = []
    list_factor_7 = []
    list_factor_8 = []
    list_factor_9 = []
    list_factor_10 = []
    list_factor_11 = []
    list_factor_12 = []
    list_factor_13 = []
    list_factor_14 = []
    
    # initialization: scores
    list_score_1 = []
    list_score_2 = []
    list_score_3 = []
    list_score_4 = []
    list_score_5 = []
    list_score_6 = []
    list_score_7 = []
    list_score_8 = []
    list_score_9 = []
    list_score_10 = []
    list_score_11 = []
    list_score_12 = []
    list_score_13 = []
    list_score_14 = []


    list_factors = [list_factor_1, list_factor_2, list_factor_3, list_factor_4, list_factor_5, 
                    list_factor_6, list_factor_7, list_factor_8, list_factor_9, list_factor_10,
                    list_factor_11, list_factor_12, list_factor_13, list_factor_14
                   ]
    list_scores = [list_score_1, list_score_2, list_score_3, list_score_4, list_score_5, 
                    list_score_6, list_score_7, list_score_8, list_score_9, list_score_10,
                    list_score_11, list_score_12, list_score_13, list_score_14
                   ]

    for index, row in df_init_explainer.iterrows():
        _temp = row.sort_values(ascending=False)
        # temporary df for getting each factor as a column with each score
        df_temp = pd.DataFrame(_temp).reset_index().rename(columns={'index':'Columns', index: 'Score'})

        # absolute value for each score
        df_temp['Score'] = df_temp['Score'].apply(lambda x: abs(x))
        df_temp.sort_values(by='Score',ascending=False, inplace=True)
        df_temp = df_temp.reset_index(drop=True)

        # get columns and scores
        x=0
        for index_in, row_in in df_temp.iterrows():
            list_factors[x].append(row_in['Columns'])
            list_scores[x].append(row_in['Score'])
            x=x+1

    # final explainer
    df_final_explainer = pd.DataFrame()
    df_final_explainer['Factor_1'] = list_factor_1
    df_final_explainer['Score_1'] = list_score_1
    df_final_explainer['Factor_2'] = list_factor_2
    df_final_explainer['Score_2'] = list_score_2
    df_final_explainer['Factor_3'] = list_factor_3
    df_final_explainer['Score_3'] = list_score_3
    df_final_explainer['Factor_4'] = list_factor_4
    df_final_explainer['Score_4'] = list_score_4
    df_final_explainer['Factor_5'] = list_factor_5
    df_final_explainer['Score_5'] = list_score_5
    df_final_explainer['Factor_6'] = list_factor_6
    df_final_explainer['Score_6'] = list_score_6
    df_final_explainer['Factor_7'] = list_factor_7
    df_final_explainer['Score_7'] = list_score_7
    df_final_explainer['Factor_8'] = list_factor_8
    df_final_explainer['Score_8'] = list_score_8
    df_final_explainer['Factor_9'] = list_factor_9
    df_final_explainer['Score_9'] = list_score_9
    df_final_explainer['Factor_10'] = list_factor_10
    df_final_explainer['Score_10'] = list_score_10
    df_final_explainer['Factor_11'] = list_factor_11
    df_final_explainer['Score_11'] = list_score_11
    df_final_explainer['Factor_12'] = list_factor_12
    df_final_explainer['Score_12'] = list_score_12
    df_final_explainer['Factor_13'] = list_factor_13
    df_final_explainer['Score_13'] = list_score_13
    df_final_explainer['Factor_14'] = list_factor_14
    df_final_explainer['Score_14'] = list_score_14
    
    return df_final_explainer

