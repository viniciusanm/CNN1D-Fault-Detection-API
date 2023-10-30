import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import SessionState
import plotly.express as px
import plotly.graph_objects as go


st.write("""
# Fault Detection App

This app receives a three phase electrical signal as input to predicts faults in MIT's! 
""")

st.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.file_uploader(label="Upload your input CSV file", 
                                 type=["csv"])


# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)



# Loading CNN model
cnn_model = tf.keras.models.load_model('saved_model/my_model')



    # Create logic for app flow
if not uploaded_file:
    st.warning("Please upload a csv file.")
    st.stop()
else:
    input_df = np.array(pd.read_csv(uploaded_file, header=None))
    if input_df.shape[0] == 384:
        input_df = input_df.reshape(1,384)

    else:
        input_df = input_df.reshape(input_df.shape[0],384)

    # Displays the user input features
    st.subheader("Three Phases of Electric Current Signal")

    # Plotly Current Signal    
    x=range(len(input_df[0, :128]))

    fig1 = px.line(x=x, y=input_df[0, :128], color_discrete_sequence=['blue'])
    fig2 = px.line(x=x, y=input_df[0, 128:256], color_discrete_sequence=['red'])
    fig3 = px.line(x=x, y=input_df[0, 256:], color_discrete_sequence=['green'])

    plot = go.Figure(data = fig1.data + fig2.data + fig3.data)

    st.plotly_chart(plot)



    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    # Apply model to make predictions
    prediction_proba = (pd.DataFrame(cnn_model.predict(input_df))).round(decimals = 3)
    prediction = prediction_proba.idxmax(axis=1)


    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")