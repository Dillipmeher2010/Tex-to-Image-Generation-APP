import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model (for demonstration)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (e.g., for MNIST 28x28 images)
        self.fc2 = nn.Linear(128, 10)   # Output layer (10 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Streamlit UI
st.title("Simple PyTorch Model in Streamlit")

st.write(
    "This app demonstrates the use of a simple PyTorch model in a Streamlit app. "
    "You can interact with it to see PyTorch in action."
)

# Button to create a random input and pass it through the model
if st.button("Run Model on Random Input"):
    model = SimpleModel()
    
    # Create a random input tensor (e.g., for MNIST images)
    random_input = torch.randn(1, 784)  # Batch size of 1, input size of 784
    output = model(random_input)  # Run the input through the model
    
    st.write(f"Model output: {output}")

# Display basic PyTorch info
st.subheader("PyTorch Information")
st.write(f"PyTorch version: {torch.__version__}")
