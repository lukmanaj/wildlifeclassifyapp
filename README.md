# Wildlife Animal Prediction App

## Overview

This repository houses a machine learning application designed for wildlife animal prediction from images. Built with Streamlit, PyTorch, and torchvision, it features a custom-trained DenseNet-201 model capable of identifying four specific animals: buffalo, elephant, rhino, and zebra. This application offers an interactive way to apply deep learning for wildlife recognition.

Live app: [Wildlife Animal Prediction App](https://wildlifeclassify.streamlit.app/)

## Dataset

The model was trained using the "African Wildlife" dataset available on Kaggle. This dataset includes images of buffalo, elephant, rhino, and zebra, providing a diverse training set for the model.

Kaggle Dataset: [African Wildlife Dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)

## Features

- **Image Upload**: Users can upload images for prediction.
- **Animal Prediction**: Identifies whether the uploaded image is of a buffalo, elephant, rhino, or zebra.
- **Confidence Score**: Provides a probability score reflecting the model's confidence in its prediction.

## Installation

Follow these steps to get the app running on your local machine:

### **Clone the Repository**

```bash
   git clone https://github.com/yourusername/wildlife-prediction-app.git
   cd wildlife-prediction-app
```

### **Environment Setup**

Windows:

```bash
    python -m venv venv
    .\venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
Install Dependencies
```

```bash
pip install -r requirements.txt
Run the App
```

```bash
streamlit run streamlit-app.py
```

## **Usage**

To use the app, either navigate to the live version or run it locally. Upload an image of one of the specified animals, and click "Predict" to see the result. The app will display the prediction and the confidence level of the model.

## **Model**
   
The pretrained DenseNet-201 model, adapted for our specific classification task, is included in this repository. It has been fine-tuned with the aforementioned Kaggle dataset to accurately distinguish between the four animal types.

## **Contributing**

Contributions are welcome. To contribute:

- Fork the repository
- Create a feature branch (git checkout -b new-feature)
- Commit your changes (git commit -am 'Add some feature')
- Push to the branch (git push origin new-feature)
- Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the creators of the "African Wildlife" dataset on Kaggle for providing the images used to train our model.
PyTorch and torchvision for the deep learning architecture and pre-trained models.
Streamlit for enabling the rapid development of interactive web applications.
Contact
For any questions, feedback, or contributions, please open an issue in the GitHub repository or contact me directly at `lukman.j.aliyu@gmail.com`
