import streamlit as st

# Dependency check
try:
    import torch
    from diffusers import StableDiffusionPipeline
    st.success("All dependencies loaded successfully!")
except ImportError as e:
    st.error(f"Import error: {e}")

# Proceed with app code below this check
# (e.g., model loading, image generation, etc.)
