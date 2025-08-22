import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL = tf.keras.models.load_model(r"C:\Users\Sakthi\Potato leaf\models\1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
st.set_page_config(page_title="Potato Disease Classification",layout="centered")
st.title("Potato Disease Classification")
st.write("Upload a potato leaf image to predict if it's Healthy, EarlyBlight, or Late Blight.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",use_column_width=True)
    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.bar_chart(predictions[0])
    st.markdown("---")
    st.caption("Built with Streamlit and TensorFlow")