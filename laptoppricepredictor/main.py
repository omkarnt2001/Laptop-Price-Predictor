import streamlit as st
import pickle
import numpy as np
import os  # Import the os module

st.title("Laptop Price Predictor")

# Load the pre-trained model and data
pipe = pickle.load(open('./laptoppricepredictor/pipe.pkl', 'rb'))  # Your pre-trained model
df = pickle.load(open('./laptoppricepredictor/data.pkl', 'rb'))   # The dataset


# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input("weight of the laptop")

# Touchscreen
touchscreen = st.selectbox('TouchScreen', ['NO', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['NO', 'Yes'])

# Screen size
Screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', 
                                                '3200x1800','2880x1800', '2560x1600', '2560x1440', 
                                                '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# hardware
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# type of OS
os = st.selectbox('Operating System', df['os'].unique())

# Predict the price when the button is clicked
if st.button('Predict Price'):

    # Convert touchscreen and ips to numeric values
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate the pixels per inch (PPI) for the display
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5 / Screen_size

    # Create the query array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, 12)

    # Make the prediction and display the result
    predicted_price = np.exp(pipe.predict(query))  # Apply the inverse of log transformation
    st.title("Predicted Price is ₹ " + str(int(predicted_price[0])))
