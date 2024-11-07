import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Streamlit app title
st.title("Image Classification with Hugging Face and PyTorch")

# File uploader to upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Model settings
MODEL_NAME = "google/vit-base-patch16-224-in21k"

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model and feature extractor
    try:
        st.write("Loading model...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        
        # Preprocess the image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get the predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        
        st.write(f"Predicted Class: {predicted_class}")

    except Exception as e:
        st.write(f"Error: {e}")
