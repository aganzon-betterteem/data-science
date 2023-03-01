import streamlit as st  
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Set Options for display
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.2f}'.format

from datetime import date
from datetime import datetime
today = date.today()

from data_processing_employees import *
from model_explainer_employees import *
from model_performance_employees import *

loaded_model = pickle.load(open('model_2023_02_27.sav', 'rb'))
explainer = pickle.load(open('explainer_2023_02_27.sav', 'rb'))
feature_importance = pickle.load(open('feature_importance_2023_02_27.sav', 'rb'))

df_employees = pd.read_excel('Integra Employee Sample Data.xlsx')
df_active = df_employees[df_employees['Exit Date'].isnull()].copy()
df_active_original = df_active.copy()

st.write("""
# Attrition Prediction Dashboard

Attrition Prediction based on Betterteem Predictive Diagram

""")

################## Date Range ##################

num_of_months = st.slider("Pick month range:", 0, 24, 6)
max_month_converted_days = num_of_months*30


################## Model Prediction ##################

df_final_result = pd.DataFrame()
while num_of_months >= 0:
    # feature selection
    
    df_active_original = df_active_original[['EEID', 'Full Name', 'Job Title', 'Department', 'Employee Group',
       'Gender', 'Marital Status', 'Education', 'Age', 'Hire Date',
       'Annual Salary', 'Daily Rate', 'Hourly Rate', 'Bonus %',
       'Salary Hike %', 'Country', 'City', 'Distance Fom Home (km)', 'Travel',
       'Satisfaction', 'Separation Type', 'Exit Date']].copy()
    df_active = df_active_original.copy()
    # processed: numerical representation
    df_active_processed = data_processing_sprout_employees(df_active)
    # predict: preparation - drop Status
    df_active_predict = df_active_processed.drop(["Status"], axis=1)
    
    # tenureship 
    df_active_predict['Tenureship'] = df_active_predict['Tenureship']+max_month_converted_days-num_of_months*30
        
    # predict: status
    temp_predicted_status = loaded_model.predict(df_active_predict)
    # print(temp_predicted_status)
    df_active_original['Predicted Status'] = temp_predicted_status
    le_status = {0: 'active', 1: 'will churn'}
    df_active_original['Predicted Status'] = df_active_original['Predicted Status'].map(le_status)
    
    # predict: distribution score
    predicted_probability_status = loaded_model.predict_proba(df_active_predict)
    df_active_original[['Will Stay Active','Will Churn']] = predicted_probability_status
    
    # tag: explanation for factors
    shap_values = explainer(df_active_predict)
    dt_explainer = pd.DataFrame(shap_values.values, columns=df_active_predict.columns)
    df_dt_init_explainer = model_explanation_initialization(dt_explainer, explainer)
    df_dt_final_explainer = model_explainer_transpose(df_dt_init_explainer)

    df_active_original.reset_index(drop=True,inplace=True)
    df_dt_final_explainer.reset_index(drop=True,inplace=True)
    df_active_original = pd.concat([df_active_original, df_dt_final_explainer],axis=1)
    
    if num_of_months==0:
        df_final_result = pd.concat([df_final_result,df_active_original],axis=0)
        df_final_result.reset_index(drop=True,inplace=True)
    else:
        # slice: active & churn
        df_churn_temp = df_active_original[df_active_original['Predicted Status']=='will churn'].copy()
        df_active_original = df_active_original[df_active_original['Predicted Status']=='active'].copy()
        df_active_original.reset_index(drop=True,inplace=True)
        
        # concatenate: empty dataframe for storage
        df_final_result = pd.concat([df_final_result,df_churn_temp])
        df_final_result.reset_index(drop=True,inplace=True)

    num_of_months = num_of_months-1

# feature engineering: tenureship
df_final_result['Tenureship'] = df_final_result.apply(lambda x: (datetime.now() - x['Hire Date']).days  if pd.isnull(x['Exit Date']) else (x['Exit Date'] - x['Hire Date']).days, axis=1)
df_final_result['Tenureship'].fillna(0, inplace=True)
df_final_result['Tenureship'] = df_final_result['Tenureship']+max_month_converted_days

# feature engineering: tenure range
df_final_result['Tenureship_Range'] = df_final_result['Tenureship'].apply(lambda x: '< 1 year' if x<365 
                                                    else '1-3 years' if x>=365 and x<1095
                                                    else '3-5 years' if x>=1095 and x<1825
                                                    else '5-10 years' if x>=1825 and x<3650
                                                    else '10 years')

# feature engineering: age banding
df_final_result['Age Banding'] = df_final_result['Age'].apply(lambda x: 'Gen Z' if x<=22 
                                                    else 'Zillenial' if x>22 and x<=26
                                                    else 'Millenial' if x>26 and x<=42
                                                    else 'Gen X' if x>42 and x<=58
                                                    else 'Boomers' if x>58 and x<=68
                                                    else 'Others')

################## Attrition Breakout ##################
st.write("""
# 
### Attrition Breakout
""")

total = df_final_result.shape[0]
likely_to_stay = round(len(df_final_result[df_final_result['Predicted Status']=='active'])/total*100,2)
likely_to_turnover = round(len(df_final_result[df_final_result['Predicted Status']!='active'])/total*100,2)

data=[likely_to_stay,likely_to_turnover]
label=['Likely to stay', 'Likely to turnover']

fig1, ax1 = plt.subplots()
plt.pie(data, labels=label, colors=['green','red'], autopct='%.0f%%')

# inner circle color
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

st.pyplot(fig1)


total_churn = df_final_result[df_final_result['Predicted Status']!='active'].shape[0]
major_BU_label = df_final_result[df_final_result['Predicted Status']!='active']['Department'].value_counts()[:1].index[0]
major_BU_percent = round(df_final_result[df_final_result['Predicted Status']!='active']['Department'].value_counts()[:1][0]/total_churn*100,2)

st.write("""
#### ! If no action is taken
""")

st.write("{}% will be coming from {}.".format(major_BU_percent,major_BU_label))
st.write("Attrition will be driven by {}.".format(feature_importance['Factors'][0]))

################## By Business Unit ##################
st.write("""
# 
### By Business Unit
""")

fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ax = df_final_result[df_final_result['Predicted Status']!='active']['Department'].value_counts().sort_values().plot(kind='barh', color=['green','blue','yellow','orange','red'])
st.pyplot(fig)

################## By Tenure Range ##################
st.write("""
# 
### By Tenure Range
""")

less_than_a_year_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Tenureship_Range']=='< 1 year')].shape[0]
one_year_to_three_years_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Tenureship_Range']=='1-3 years')].shape[0]
three_to_five_years_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Tenureship_Range']=='3-5 years')].shape[0]
five_to_ten_years_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Tenureship_Range']=='5-10 years')].shape[0]
ten_years_over_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Tenureship_Range']=='10 years')].shape[0]

y=[less_than_a_year_count,one_year_to_three_years_count,three_to_five_years_count,five_to_ten_years_count,ten_years_over_count]
x=['< 1 year','1-3 years','3-5 years','5-10 years','> 10 years']
fig2, ax2 = plt.subplots()
plt.bar(x,y,color=['yellow','orange','red','blue','green']) 
st.pyplot(fig2)

################## By Age Banding ##################
st.write("""
# 
### By Age Banding
""")

gen_z_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Age Banding']=='Gen Z')].shape[0]
zillenial_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Age Banding']=='Zillenial')].shape[0]
millenial_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Age Banding']=='Millenial')].shape[0]
gen_x_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Age Banding']=='Gen X')].shape[0]
boomers_count = df_final_result[(df_final_result['Predicted Status']!='active') & (df_final_result['Age Banding']=='Boomers')].shape[0]

total_active = df_final_result[df_final_result['Predicted Status']!='active'].shape[0]
gen_z_percentage = round(gen_z_count/total_active*100,2)
zillenial_percentage = round(zillenial_count/total_active*100,2)
millenial_percentage = round(millenial_count/total_active*100,2)
gen_x_percentage = round(gen_x_count/total_active*100,2)
boomers_percentage = round(boomers_count/total_active*100,2)

b=[gen_z_percentage,zillenial_percentage,millenial_percentage,gen_x_percentage,boomers_percentage]
a=['Gen Z','Zillenial','Millenial','Gen X','Boomers']
fig3, ax3 = plt.subplots()
plt.bar(a,b, color=['yellow','orange','red','blue','green']) 
st.pyplot(fig3)

################## By Job Role ##################
st.write("""
# 
### By Job Role
""")
fig4, ax4 = plt.subplots()
fig4.patch.set_facecolor('white')
df_final_result[df_final_result['Predicted Status']!='active']['Job Title'].value_counts().sort_values().plot(kind='bar', color=['green'])
st.pyplot(fig4)

################## By Gender ##################
st.write("""
# 
### By Gender
""")
total = df_final_result.shape[0]
male_count = round(len(df_final_result[df_final_result['Gender']=='Male'])/total*100,2)
female_count = round(len(df_final_result[df_final_result['Gender']=='Female'])/total*100,2)
lqbtqia_count = round(len(df_final_result[(df_final_result['Gender']!='Male') &
                                         (df_final_result['Gender']!='Female')])/total*100,2)

data_gender=[male_count,female_count, lqbtqia_count]
label_gender=['He/Him', 'She/Her', 'They/Them']

fig5, ax5 = plt.subplots()
plt.pie(data_gender, labels=label_gender, colors=['blue','pink', 'yellow'], autopct='%.0f%%')

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.legend(data_gender, loc="best")
st.pyplot(fig5)

################## Drivers for Attrition ##################
st.write("""
# 
### Drivers for Attrition
""")
feature_importance['Importance'] = feature_importance['Importance'].apply(lambda x: x*100)
st.table(data=feature_importance)

################## Sample Employee Per Row ##################
st.write("""
# 
### Sample Employee Per Row
""")
st.table(data=df_final_result.head())