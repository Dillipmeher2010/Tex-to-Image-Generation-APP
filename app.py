import streamlit as st
from diffusers import StableDiffusionPipeline
from huggingface_hub import cached_download

# Ensure that model and other components are downloaded
model_name = "CompVis/stable-diffusion-v-1-4-original"
cache_dir = cached_download(model_name)

# Load the Stable Diffusion model from the cache
@st.cache_resource  # To prevent reloading the model every time
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=cache_dir)
    return pipe

# Load the model
pipe = load_model()

# Streamlit interface for input
st.title("Text-to-Image Generation with Stable Diffusion")
prompt = st.text_input("Enter a prompt:", "A futuristic city at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
