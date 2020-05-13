import streamlit as st 
from PIL import Image
import numpy as np
import cv2
import PIL
import numpy

st.title("Let's Classify Image")

def init_classify():
    global net, classes
    rows = open('Model/synset_words.txt').read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    weight = 'Model/bvlc_googlenet.caffemodel'
    arch = 'Model/bvlc_googlenet.prototxt'
    net = cv2.dnn.readNetFromCaffe(arch, weight)

def predict(image1):
    init_classify()
    pil_image = PIL.Image.open(image1).convert('RGB')
    open_cv_image = numpy.array(pil_image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy() 	
    #image = cv2.imread(open_cv_image)
    # Make prediction
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224),
                                                (104, 117, 123)))
    net.setInput(blob)
    prediction = net.forward()
    idxs = np.argsort(prediction[0])[::-1][0]
    label = classes[idxs]
    confidence = round(prediction[0][idxs] * 100, 2)
    #comb = label + "," + str(confidence)
    return label,confidence

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


