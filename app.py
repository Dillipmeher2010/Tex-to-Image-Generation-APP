import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Load model on CPU for compatibility with Streamlit Cloud
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Use CPU instead of GPU

# Function to generate images with specified parameters
def generate_image(prompt, params):
    image = pipe(prompt, **params).images[0]
    return image

# Streamlit UI setup
st.title("Text-to-Image Generation with Stable Diffusion")

# Prompt input
prompt = st.text_area("Enter your prompt:", "a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin")

# Parameter inputs
num_inference_steps = st.slider("Number of inference steps:", 1, 200, 100)
width = st.slider("Image width:", 128, 1024, 512)
height = st.slider("Image height:", 128, 1024, 768)
num_images_per_prompt = st.slider("Number of images:", 1, 4, 1)
negative_prompt = st.text_input("Negative prompt:", "ugly, distorted, low quality")

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        params = {
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "num_images_per_prompt": num_images_per_prompt,
            "negative_prompt": negative_prompt
        }
        image = generate_image(prompt, params)
        
        # Display image
        st.image(image, caption="Generated Image", use_column_width=True)
