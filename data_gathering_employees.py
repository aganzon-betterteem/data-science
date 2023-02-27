#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def data_gathering_sprout_employees_sample(file_name='Integra Employee Sample Data.xlsx'):
    """
    Description: Data gathering of sample employee data from Sprout
    ---
    Parameters:
    file_name: Str
    ---
    returns df: DataFrame
    """
    df = pd.read_excel(file_name)
    return df

