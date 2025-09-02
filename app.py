import streamlit as st
import pandas as pd
import numpy as np
import os
# Profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# ML
from pycaret.regression import setup as reg_setup,compare_models as  reg_compare_models,pull as reg_pull,save_model as reg_save_model
from pycaret.classification import setup as class_setup,compare_models as class_compare_models,pull as class_pull,save_model as class_save_model
########################################
df=None

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col=None, header=0)

###
def regressor(target):
    df[target]=df[target].astype("float")
    st.info("Regression Modeling has started")
    reg_setup(
            data=df,
            target=target,
            session_id=123,
            verbose=False  # suppresses logs if you want
            )
    s=reg_pull()
    st.info("This is the ML Experimentation results")
    st.dataframe(s)
    best_model=reg_compare_models()
    models=reg_pull()
    st.info("This is the Experimentation model")
    st.dataframe(models.astype(str))  # convert everything to string
    reg_save_model(best_model,"reg_best_model")


def classifier(target):
    st.info("Classifier Modeling has started")
    class_setup(
            data=df,
            target=target,
            session_id=123,
            verbose=False  # suppresses logs if you want
            )
    s=class_pull()
    st.info("This is the ML Experimentation results")
    st.dataframe(s)
    best_model=class_compare_models()
    models=class_pull()
    st.info("This is the Experimentation model")
    st.dataframe(models.astype(str))  # convert everything to string
    class_save_model(best_model,"classif_best_model")

with st.sidebar:
    st.image("automl.webp")
    st.title("AutoML Streamlit App")
    choice=st.radio("Choose an option",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline with streamlit.")
    
if choice=="Upload":
    st.title("Upload your dataset for modeling")
    file=st.file_uploader("Upload your dataset")
    if file:
        df=pd.read_csv(file,index_col=None)
        st.dataframe(df.head())
        df.to_csv("source_data.csv",index=None)
        
if choice=="Profiling":
    st.title("Exploratory Data Analysis")
    if df is None:
        st.info("Please upload a dataset first")
    else:    
        profiling=ProfileReport(df,title="Pandas Profiling Report",explorative=True)
        st_profile_report(profiling)       
    
if choice=="ML":
    st.title("Machine Learning Model")
    if df is None:
        st.info("Please upload a dataset first")
    else:    
        problem_type=st.selectbox("Select the problem type",["Classification","Regression"])
        target=st.selectbox("Select your target variable",df.columns)
        if st.button("Start Modeling"):
            if problem_type=="Regression":
                regressor(target)
            else:
                classifier(target)

if choice=="Download":
    st.title("Download the best model")
    if not os.path.exists("best_model.pkl"):
        st.info("Please build a model first")
    else :
        with open("best_model.pkl","rb") as f:
            st.download_button("Click here to download the model",f,"best_model.pkl")
