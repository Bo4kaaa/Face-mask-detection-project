import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("face_mask_detection3.h5")
labels = ["Mask", "No Mask"]

def predict_mask(image):
    img = image.convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])  
    return {"Mask": prob, "No Mask": 1 - prob}

interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="😷 Face Mask Detection",
    description="Upload an image to check if the person is wearing a mask or not."
)

interface.launch()