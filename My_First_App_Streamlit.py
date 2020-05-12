import streamlit as st 
from PIL import Image
from classify import predict
from classify import init_classify
import numpy as np
st.title("Let's Classify Image")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)	
    st.image(image, caption='Uploaded Image.')
    st.write("")
    if st.button("Click Here to Predict"):
        st.write('\n','\n')
        st.write("Classifying...")
        label,confidence = predict(uploaded_file)
        st.write('Our Model Predict the:**',label,' **with **',confidence,'** confidence')


