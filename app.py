import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import tensorflow as tf

# ----------------- Load TFLite Model -----------------
@st.cache_resource
def load_my_model():
    interpreter = tf.lite.Interpreter(model_path="../saved_model/visual_check.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_my_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------- Preprocess Image -----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# ----------------- Make Prediction -----------------
def predict_image(img_np):
    input_data = preprocess_image(img_np)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    pred = output_data[0][0]
    label = "DEFECT" if pred >= 0.5 else "GOOD"
    confidence = float(pred if label == "DEFECT" else 1 - pred)
    return label, confidence

# ----------------- Log Result -----------------
def log_result(source, label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, source, label, confidence]], columns=["Timestamp", "Source", "Label", "Confidence"])
    df.to_csv("visual_check_log.csv", mode='a', header=not pd.io.common.file_exists("visual_check_log.csv"), index=False)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Visual Quality Check", layout="centered")
st.title("📦 Visual Quality Check System")
st.caption("Use webcam or upload an image to classify product quality.")

tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Webcam Capture"])

# -------- Upload Image Tab --------
with tab1:
    if "upload_mode" not in st.session_state:
        st.session_state.upload_mode = "start"
        st.session_state.uploaded_file = None

    if st.session_state.upload_mode == "start":
        uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"], key="upload1")
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.upload_mode = "show"

    if st.session_state.upload_mode == "show" and st.session_state.uploaded_file is not None:
        st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"], key="upload2", label_visibility="collapsed")
        img = Image.open(st.session_state.uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        label, confidence = predict_image(img_np)
        st.success(f"Prediction: **{label}**")
        st.progress(confidence)
        st.metric("Confidence Score", f"{confidence:.2f}")

        if st.button("📄 Log Result"):
            log_result("Uploaded Image", label, confidence)
            st.info("Result logged to CSV.")

        if st.button("🔄 Try Another Image"):
            st.session_state.upload_mode = "start"
            st.session_state.uploaded_file = None

# -------- Webcam Capture Tab --------
with tab2:
    if "webcam_mode" not in st.session_state:
        st.session_state.webcam_mode = "start"
        st.session_state.webcam_picture = None

    if st.session_state.webcam_mode == "start":
        picture = st.camera_input("Take a picture")
        if picture:
            st.session_state.webcam_picture = picture
            st.session_state.webcam_mode = "show"

    if st.session_state.webcam_mode == "show" and st.session_state.webcam_picture is not None:
        img = Image.open(st.session_state.webcam_picture).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Captured Image", use_container_width=True)

        label, confidence = predict_image(img_np)
        st.success(f"Prediction: **{label}**")
        st.progress(confidence)
        st.metric("Confidence Score", f"{confidence:.2f}")

        if st.button("📄 Log Result (Webcam)"):
            log_result("Webcam", label, confidence)
            st.info("Result logged to CSV.")

        if st.button("🔁 Recapture Image"):
            st.session_state.webcam_mode = "start"
            st.session_state.webcam_picture = None
