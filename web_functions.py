"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('milk.csv')

    # Perform feature and target split
    X = df[["Size_of_ROIs","Mean_Value","St_Deviation","Skewness","Gradient_Mean_value","Gradient_Variance","Percentage_non_zero_Gradients","Contrast","Entropy","Homogenity","Run_Percentage","Long_run_Emphasis","Gray_Value_Distribution"]]
    y = df['Yearly_Production']

    return df, X, y

@st.cache_data()
def train_model(X, y):
    y = np.array(y).copy()
    """This function trains the model and return the model and model score"""
    # Create the model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)

    # Return the values
    return model, score

def predict(X, y, features):
    
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
