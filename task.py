import streamlit as st
import os
from fastai.vision.all import *

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

# Load the model
learn_inf = load_learner(model_path)

st.title("Image Classification App")
st.write("Upload an image and the app will predict the corresponding label.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")