import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

# Load your pre-trained model (replace "model2.h5" with the actual model file path)
model = tf.keras.models.load_model("model2.h5")

# File uploader widget
uploaded_file = st.file_uploader("Choose a plant image (JPG format)", type=["jpg", "jpeg"])

# Dictionary mapping class indices to disease labels based on your model's predictions
map_dict = {
    0: 'Healthy',
    1: 'Powdery',
    2: 'Rust',
}

if uploaded_file is not None:
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (64, 64))
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    Generate_pred = st.button("Generate Prediction")
    if Generate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict[prediction]))
