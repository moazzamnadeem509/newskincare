import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model_path = hf_hub_download(
        repo_id="Moazzam56/skincare",
        filename="best_model.keras"
    )

    model = tf.keras.models.load_model(model_path)

    with open("class_names.json") as f:
        meta = json.load(f)

    return model, meta

model, meta = load_assets()
class_names = meta["class_names"]
img_size = tuple(meta["img_size"])

st.title("🩺 Skin Cancer Detection")
st.caption("Model: " + meta["model_name"])
st.warning("Educational demo only — NOT a medical diagnosis. Consult a dermatologist.")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = np.asarray(
        image.resize(img_size),
        dtype=np.float32
    )[None]

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))

    st.subheader("Prediction: " + class_names[pred].upper())
    st.metric("Confidence", f"{probs[pred]:.1%}")
    st.bar_chart({
        c: float(p)
        for c, p in zip(class_names, probs)
    })
