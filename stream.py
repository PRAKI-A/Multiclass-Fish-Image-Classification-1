


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.preprocessing.image import img_to_array

##### Background Styling
def set_sea_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");         
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #ffffff;
        }
        .block-container {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 12px;
        }
        .centered-title h1 {
            text-align: center;
            font-size: 2.5em;
            color: #ffffff;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_sea_background()
st.markdown('<div class="centered-title"><h1>🐟 Fish Classifier</h1></div>', unsafe_allow_html=True)

##### Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("inception_model.h5")

model = load_model()

##### Detect input size automatically
try:
    img_height, img_width = model.input_shape[1:3]
    img_size = (img_width, img_height)
except Exception:
    img_size = (224, 224)  # fallback safe default

##### Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

st.markdown("### 📤 Upload a fish image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        image = image.resize(img_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0   # safe normalization if we don’t know the exact backbone

        # Predict
        preds = model.predict(image_array)
        predicted_index = np.argmax(preds)
        predicted_class = idx_to_class.get(predicted_index, f"Class {predicted_index}")
        confidence = float(np.max(preds)) * 100

        ### Display results
        st.markdown(
            f"""
            <div style='
                background-color: #fff3e0;
                padding: 15px;
                border-left: 5px solid #4CAF50;
                border-radius: 8px;
                color: #4CAF50;
                font-size: 18px;
                font-weight: bold;
            '>
            🎯 Predicted Fish: {predicted_class} ({confidence:.2f}% confidence)
            </div>
            """,
            unsafe_allow_html=True
        )

        #### Top 3 predictions
        top_3_indices = preds[0].argsort()[-3:][::-1]
        st.subheader("🔍 Top-3 Predictions:")
        for i in top_3_indices:
            st.write(f"- {idx_to_class.get(i, f'Class {i}')}: {preds[0][i]*100:.2f}%")

        st.markdown("---")
        st.markdown("📌 Model: Custom CNN | Built with Streamlit | Fish dataset")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
