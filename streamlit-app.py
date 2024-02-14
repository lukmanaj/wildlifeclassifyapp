import streamlit as st
import torch
import torchvision
from torchvision import transforms

# Assuming necessary imports for your model to work are included
# such as torchvision.models, torch.nn, etc.

def load_model(model_path, device):
    weights = torchvision.models.DenseNet201_Weights.DEFAULT # best available weight
    model = torchvision.models.densenet201(weights=weights).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1920, out_features=4, bias=True)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)), # Adjust size to match model's expected input
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_prediction(model, image, device):
    class_names = ['buffalo', 'elephant', 'rhino', 'zebra']
    image = image.unsqueeze(0).to(device) # Add batch dimension and move to device
    with torch.no_grad():
        pred_logits = model(image)
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
    return class_names[pred_label], pred_prob.max().item()

# Streamlit app starts here
st.title("Wildlife Animal Prediction App")

# Sidebar for model path - optional if model path is fixed
# model_path = st.sidebar.text_input("Model Path", value='model/densenetafri.pth')
model_path = 'model/densenetafri.pth'  # Fixed model path

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = torchvision.io.read_image(uploaded_file).type(torch.float32) / 255.0
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict button
    if st.button('Predict'):
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model = load_model(model_path, device)

        # Preprocess and predict
        preprocessed_image = preprocess_image(image)
        prediction, probability = get_prediction(model, preprocessed_image, device)

        # Display prediction
        st.write(f"Prediction: {prediction}, Probability: {probability:.3f}")

# Note: Adjust the preprocess_image function according to the model's requirement.
