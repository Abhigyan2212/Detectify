import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

st.title("Object Detection Web App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    results = model.predict(image)
    st.image(results[0].plot())
