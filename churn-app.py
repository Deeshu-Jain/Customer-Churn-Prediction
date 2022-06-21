import streamlit as st

import pandas as pd

import numpy as np

import pickle

import base64

import seaborn as sns

import matplotlib.pyplot as plt

st.write("""

# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with the company. 

This app predicts the probability of a customer churning using Telco Customer data. Here customer churn means the customer does not make another purchase after a period of time. 

""")

df_selected=pd.read_csv("telco.csv")

df_selected_all=df_selected[['gender', 'Partner','InternetService','Contract',
'Dependents', 'PhoneService', 
'tenure', 'MonthlyCharges', 'Churn']].copy()

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def user_input_features():
    gender = st.sidebar.selectbox('gender',('Male','Female'))
    PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
    tenure = st.sidebar.slider('tenure', 0.0,72.0, 0.0)
    
    data = {'gender':[gender],
            'PaymentMethod':[PaymentMethod], 
            'MonthlyCharges':[MonthlyCharges], 
            'tenure':[tenure],}
    features = pd.DataFrame(data)
    
    return features

if uploaded_file is not None:

    input_df = pd.read_csv(uploaded_file)

else:
    
    input_df = user_input_features()

churn_raw = pd.read_csv('telco.csv')

churn_raw.fillna(0, inplace=True)

churn = churn_raw.drop(columns=['Churn'])

df = pd.concat([input_df,churn],axis=0)

encode = ['gender','PaymentMethod']

for col in encode:

    dummy = pd.get_dummies(df[col], prefix=col)

    df = pd.concat([df,dummy], axis=1)

    del df[col]

df = df[:1] # Selects only the first row (the user input data)

#Select the features we want to display:

features = ['tenure','MonthlyCharges','gender_Female','gender_Male',
       'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
       'PaymentMethod_Mailed check']

df = df[features]

#Finally, we display the default input using the write method:

# Displays the user input features

st.subheader('User Input features')

print(df.columns)

if uploaded_file is not None:

    st.write(df)

else:

    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')

    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])
st.write(churn_labels[prediction])

if churn_labels[prediction]=='No':
    st.write("Customer will not make any other purchase for some time period")
else:   
    st.write("Customer will make some other purchase after some time.")
st.subheader('Prediction Probability')
st.write(prediction_proba)