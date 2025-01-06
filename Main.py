import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import models
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img



def model_prediction(test_image):
    #model = tf.keras.models.load_model('trained_model.h5')
    model = tf.keras.models.load_model('D:/Plant Disease Detection System/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/trained_model.h5')
    img = load_img(test_image, color_mode='rgb', target_size=(128, 128))
    # img = tf.keras.preprocessing.image.load_img(test_image, (128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    result = np.argmax(predictions)
    return result

st.header("Plant Disease Recognition")
test_img = st.file_uploader("Choose an image")
if(st.button("Show Image")):
    st.image(test_img)
if(st.button("Predict")):
    st.write("Our Prediction")
    result = model_prediction(test_img)
    class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
    st.success("Model is Predicting it is a {}".format(class_names[result]))
