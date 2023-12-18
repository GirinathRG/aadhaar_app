import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Load your trained model
model_path = r"C:\Users\Unknown\Desktop\Aadhaar Detection\models\aadhaar.h5"
model = load_model(model_path)

def predict_aadhaar(img_array):
    # Normalize the image
    img_array /= 255.0

    # Make predictions
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Assuming your model has two classes (0: Not Aadhaar, 1: Aadhaar)
    predicted_class = np.argmax(prediction)

    # Return True if predicted as Aadhaar, False otherwise
    return predicted_class == 1

def main():
    st.title("Aadhaar Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(128, 128))
        img_array = image.img_to_array(img)

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        if predict_aadhaar(img_array):
            st.success("This is an Aadhaar card!")
        else:
            st.warning("This is not an Aadhaar card.")

if __name__ == "__main__":
    main()
