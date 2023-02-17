import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import sys

st.title('TvScriptor')
st.write('This script is a LSTM model that generates a script based on a given text file. The model is trained on the first two seasons of The Office. The model is then used to generate a script based on the first two seasons of The Office.')

#create an input box for the user to enter the text file
st.write('Enter the text file you want to use to train the model:')
text_file = st.text_input('Enter the text file you want to use to train the model:')
st.write('You entered:', text_file)

# Load the text file
with open(text_file, mode='r', encoding='utf-8') as f:
    text = f.read() 

# Create a mapping from unique characters to indices


