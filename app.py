import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Title of the Streamlit app
st.title("Image Classification using Hugging Face and PyTorch")

# Instructions for the user
st.markdown("Upload an image to classify using a pre-trained model.")

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Download a pre-trained model and its feature extractor
    model_name = "google/vit-base-patch16-224-in21k"
    st.write("Downloading model...")
    
    # Download the model and feature extractor
    model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    feature_extractor_path = hf_hub_download(repo_id=model_name, filename="preprocessor_config.json")
    
    # Load the model and feature extractor using transformers
    model = AutoModelForImageClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
    
    # Prepare image for prediction
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Predict using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    
    # Show the prediction
    st.write(f"Predicted class: {predicted_class}")
