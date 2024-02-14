import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import io

# Define the function to load the model
def load_model(model_path, device):
    weights = torchvision.models.DenseNet201_Weights.DEFAULT  # best available weight
    model = torchvision.models.densenet201(weights=weights).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1920, out_features=4, bias=True)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Define the function for preprocessing the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        
    ])
    return transform(image)

# Define the function for getting predictions
def get_prediction(model, image, device):
    class_names = ['buffalo', 'elephant', 'rhino', 'zebra']
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        pred_logits = model(image)
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
    return class_names[pred_label.item()], pred_prob.max().item()

# Streamlit app starts here
st.title("Wildlife Animal Prediction App")

uploaded_file = st.file_uploader("Upload an image of one of the following: Bufallo, Elephant, Rhino, or Zebra", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file-like object to bytes, then open it with PIL
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model
        model_path = 'model/densenetafri.pth'  # Fixed model path
        model = load_model(model_path, device)
        
        # Preprocess the image and predict
        preprocessed_image = preprocess_image(image)
        prediction, probability = get_prediction(model, preprocessed_image, device)
        
        # Display the prediction
        st.write(f"Prediction: {prediction}, Probability: {probability:.3f}")

