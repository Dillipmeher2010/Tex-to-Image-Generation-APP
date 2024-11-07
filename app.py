import streamlit as st

# Dependency check
try:
    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import pipeline
    st.success("All dependencies loaded successfully!")
except ImportError as e:
    st.error(f"Import error: {e}")

# Proceed with your Streamlit app code if dependencies load without issue
# The rest of your app.py code goes here, including model loading and image generation
