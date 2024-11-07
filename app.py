import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    st.info(f"CUDA available. Using {torch.cuda.get_device_name(0)}")
else:
    st.warning("CUDA not available, using CPU.")

# Streamlit app title
st.title("Text-to-Image Generation with Dreamlike Diffusion")
st.write("Generate high-quality images from text prompts using the Dreamlike Diffusion model.")

# Define the model to use
model_id = "dreamlike-art/dreamlike-diffusion-1.0"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)

# Move the model to GPU if available, else use CPU
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Text input for the prompt
prompt = st.text_area("Enter a text prompt for image generation:", "A beautiful sunset over the mountains")

# Parameters for image generation
num_inference_steps = st.slider("Number of Inference Steps", min_value=10, max_value=100, value=50)
height = st.slider("Image Height", min_value=256, max_value=1024, value=512)
width = st.slider("Image Width", min_value=256, max_value=1024, value=512)
num_images_per_prompt = st.slider("Number of Images Per Prompt", min_value=1, max_value=4, value=1)
negative_prompt = st.text_input("Negative Prompt (optional)", "ugly, distorted, low quality")

# Generate button
generate_button = st.button("Generate Image")

# Function to generate image
def generate_image(pipe, prompt, num_inference_steps, height, width, num_images_per_prompt, negative_prompt=None):
    params = {
        'num_inference_steps': num_inference_steps,
        'height': height,
        'width': width,
        'num_images_per_prompt': num_images_per_prompt
    }
    if negative_prompt:
        params['negative_prompt'] = negative_prompt

    # Generate the image(s)
    images = pipe(prompt, **params).images

    # Display the image(s)
    if len(images) > 1:
        cols = st.columns(len(images))
        for i, image in enumerate(images):
            with cols[i]:
                st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
    else:
        st.image(images[0], caption="Generated Image", use_column_width=True)

# Generate the image when the button is clicked
if generate_button:
    with st.spinner("Generating your image..."):
        generate_image(pipe, prompt, num_inference_steps, height, width, num_images_per_prompt, negative_prompt)

    st.success("Image generated successfully!")
