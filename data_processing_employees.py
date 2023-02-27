#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# date time libraries
from datetime import date
from datetime import datetime
today = date.today()


# In[2]:


def get_salary_grade(df, salary_grade):
    """
    Description: Get the salary grade of each employee based on 2022 salary grade table
    ---
    Parameters: 
    df: DataFrame
    salary_grade: str
    ---
    returns df: DataFrame
    """
    # data gathering: salary grade
    df_salary_grade = pd.read_excel(salary_grade)
    
    salary_grade_list = []
    for x in df['Annual Salary']:
        for index, row in df_salary_grade.iterrows():
            if  x >= row['Annual'] and x < df_salary_grade['Annual'][index+1]:
                salary_grade_list.append(row['Salary Grade'])
                break
            
    df['Annual Salary Grade'] = salary_grade_list
    return df


# In[3]:


def category_to_label(df):
    """
    Description: manually map the category into numerical representation
    ---
    Parameters: 
    df: DataFrame
    ---
    returns df: DataFrame
    """
    # mapping values
    le_job_title = {'Accountant I': 0, 'Accountant II': 1, 'Analyst': 2, 'Analyst II': 3, 'Associate I': 4, 'Associate II': 5, 'Automation Engineer': 6, 'Client Services': 7, 'Director': 8, 'Engineering Manager': 9, 'HRIS Analyst': 10, 'Manager': 11, 'Quality Engineer': 12, 'Sales Associate I': 13, 'Sales Associate II': 14, 'Senior Software Engineer': 15, 'Software Engineer': 16, 'Sr. Analyst': 17, 'Sr. Business Partner': 18, 'Sr. Manger': 19, 'Supervisor': 20, 'Test Engineer': 21, 'Vice President': 22}
    le_department = {'Business Development': 0, 'Engineering': 1, 'Finance': 2, 'Human Resources': 3, 'Operations': 4}
    le_employee_group = {'Call Center': 0, 'Corporate': 1, 'Shared Services': 2, 'Technology': 3}
    le_gender = {'Female': 0, 'Male': 1}
    le_marital_status = {'Married': 0, 'Single': 1, 'Solo Parent': 2, 'Widowed': 3}
    le_education = {'Graduate': 0, 'Post-Graduate': 1, 'Undergraduate': 2}
    le_country = {'Philippines': 0}
    le_city = {'Cebu': 0, 'Davao': 1, 'Manila': 2}
    le_travel = {'Non_Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2}
    le_status = {'active': 0, 'churn': 1}
    
    df['Job Title'] = df['Job Title'].map(le_job_title)
    df['Department'] = df['Department'].map(le_department)
    df['Employee Group'] = df['Employee Group'].map(le_employee_group)
    df['Gender'] = df['Gender'].map(le_gender)
    df['Marital Status'] = df['Marital Status'].map(le_marital_status)
    df['Education'] = df['Education'].map(le_education)
    df['Country'] = df['Country'].map(le_country)
    df['City'] = df['City'].map(le_city)
    df['Travel'] = df['Travel'].map(le_travel)
    df['Status'] = df['Status'].map(le_status)
    
    return df


# In[4]:


def data_processing_sprout_employees(df, salary_grade='Salary_Grade.xlsx'):
    """
    Description:
    ---
    Parameters: 
    df: DataFrame
    salary_grade: str
    ---
    returns df: DataFrame
    """
    # data cleaning
    df['Exit Date'].fillna(' ', inplace=True)
    
    # feature engineering: tenureship
    df['Tenureship'] = df.apply(lambda x: (datetime.now() - x['Hire Date']).days  if x['Exit Date'] == ' ' else (x['Exit Date'] - x['Hire Date']).days, axis=1)
    df['Tenureship'].fillna(0, inplace=True)
    # feature engineering: annual salary grade 
    df = get_salary_grade(df, salary_grade)
    # feature engineering: status
    df['Status'] = df['Exit Date'].apply(lambda x: 'active' if x == ' ' else 'churn')
    
    # label encoder: convert category into numerical representation
    df = category_to_label(df)
    
    # feature selection
    df.drop(['EEID','Full Name','Country','Exit Date','Hire Date','Separation Type','Annual Salary','Daily Rate',
             'Hourly Rate','Marital Status'],axis=1,inplace=True)

    return df

