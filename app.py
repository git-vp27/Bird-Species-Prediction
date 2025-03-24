import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ✅ Move set_page_config to the very top before any Streamlit commands
st.set_page_config(page_title="Bird Species Detection", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('densenet_bird_model.h5')  # Update with your model path
    return model

model = load_model()

# Class labels (Update with your actual bird species names)
class_labels = [
    "Asian-Green-Bee-Eater", "Brown-Headed-Barbet", "Cattle-Egret", "Common-Kingfisher",
    "Common-Myna", "Common-Rosefinch", "Common-Tailorbird", "Coppersmith-Barbet",
    "Forest-Wagtail", "Gray-Wagtail", "Hoopoe", "House-Crow",
    "Indian-Grey-Hornbill", "Indian-Peacock", "Indian-Pitta", "Indian-Roller",
    "Jungle-Babbler", "Northern-Lapwing", "Red-Wattled-Lapwing", "Ruddy-Shelduck",
    "Rufous-Treepie", "Sarus-Crane", "White-Breasted-Kingfisher", "White-Breasted-Waterhen",
    "White-Wagtail"
]

# Page title and description
st.title("🦅 Bird Species Detection App")
st.markdown("Upload an image of a bird, and the model will predict its species!")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# Process and predict if file is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize for model input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Predicted bird species
    predicted_species = class_labels[predicted_class]

    # Show results
    st.success(f"✅ Predicted Bird Species: **{predicted_species}**")
    st.info(f"🎯 Confidence: **{confidence:.2f}%**")

    # Show probability for all classes
    # st.subheader("📊 Class Probabilities")
    # for i, prob in enumerate(prediction[0]):
    #     st.write(f"**{class_labels[i]}**: {prob * 100:.2f}%")

# About section
st.sidebar.header("About")
st.sidebar.write("""
This app uses a trained deep learning model to predict the bird species from an uploaded image.
- Model: CNN / DenseNet / ResNet
- Input: Image of a bird
- Output: Predicted bird species with confidence
""")
