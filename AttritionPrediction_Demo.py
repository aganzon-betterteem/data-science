#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

# Set Options for display
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.2f}'.format


# In[2]:


from data_processing_employees import *
from model_explainer_employees import *
from model_performance_employees import *

loaded_model = pickle.load(open('model_2023_02_27.sav', 'rb'))
explainer = pickle.load(open('explainer_2023_02_27.sav', 'rb'))


# In[3]:


df_employees = pd.read_excel('Integra Employee Sample Data.xlsx')
df_active = df_employees[df_employees['Exit Date'].isnull()].copy()
df_active_original = df_active.copy()


# In[4]:


df_active_processed = data_processing_sprout_employees(df_active)
df_active_predict = df_active_processed.drop(["Status"], axis=1)


# In[5]:


# predict: status
predicted_status = loaded_model.predict(df_active_predict)
df_active_original['Predicted Status'] = predicted_status
le_status = {0: 'active', 1: 'will churn'}
df_active_original['Predicted Status'] = df_active_original['Predicted Status'].map(le_status)

# tag: explanation for factors
shap_values = explainer(df_active_predict)
dt_explainer = pd.DataFrame(shap_values.values, columns=df_active_predict.columns)
df_dt_init_explainer = model_explanation_initialization(dt_explainer, explainer)
df_dt_final_explainer = model_explainer_transpose(df_dt_init_explainer)


# In[6]:


df_active_original.reset_index(drop=True,inplace=True)
df_dt_final_explainer.reset_index(drop=True,inplace=True)
df_final_result = pd.concat([df_active_original,df_dt_final_explainer],axis=1)


# In[7]:


df_final_result


# In[8]:


tenureship = st.slider('Slider', 0, 36, 6)
tenureship


# In[ ]:




