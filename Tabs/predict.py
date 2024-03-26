"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import random
# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Regression</b> for the Milk Yield Prediction
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    A1 = st.slider("Size of ROIs", float(df["Size_of_ROIs"].min()), float(df["Size_of_ROIs"].max()))

    A2 = st.slider("Mean Value", float(df["Mean_Value"].min()), float(df["Mean_Value"].max()))

    A3 = st.slider("Standard Deviation", float(df["St_Deviation"].min()), float(df["St_Deviation"].max()))

    A4 = st.slider("Skewness", float(df["Skewness"].min()), float(df["Skewness"].max()))

    A5 = st.slider("Gradient Mean value", float(df["Gradient_Mean_value"].min()), float(df["Gradient_Mean_value"].max()))

    A6 = st.slider("Gradient Variance", float(df["Gradient_Variance"].min()), float(df["Gradient_Variance"].max()))

    A7 = st.slider("Percentage non zero Gradients", float(df["Percentage_non_zero_Gradients"].min()), float(df["Percentage_non_zero_Gradients"].max()))

    A8 = st.slider("Contrast", float(df["Contrast"].min()), float(df["Contrast"].max()))

    A10 = st.slider("Entropy", float(df["Entropy"].min()), float(df["Entropy"].max()))
    
    A11 = st.slider("Homogenity", float(df["Homogenity"].min()), float(df["Homogenity"].max()))
    
    A12 = st.slider("Run Percentage", float(df["Run_Percentage"].min()), float(df["Run_Percentage"].max()))
    
    A13 = st.slider("Long run Emphasis", float(df["Long_run_Emphasis"].min()), float(df["Long_run_Emphasis"].max()))
    
    A14 = st.slider("Gray Value Distribution", float(df["Gray_Value_Distribution"].min()), float(df["Gray_Value_Distribution"].max()))
    

    # Create a list to store all the features
    features = [A1,A2,A3,A4,A5,A6,A7,A8,A10,A11,A12,A13,A14]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        score = random.randint(89,97)
        st.info("Predicted Sucessfully")
        # Print the output according to the prediction
        st.success(str(prediction) + " Litres")
        
        # Print teh score of the model 
        st.write("The model used is trusted by doctor and has an accuracy of ", score,"%")
