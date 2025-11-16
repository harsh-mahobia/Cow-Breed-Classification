from ultralytics import YOLO
from PIL import Image
import streamlit as st
import io
import numpy as np


@st.cache_resource(show_spinner=True)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_topk_probs(result, names, k: int = 5):
    if not hasattr(result, "probs") or result.probs is None:
        return []
    probs = result.probs.data.cpu().numpy()
    topk_idx = np.argsort(probs)[::-1][:k]
    return [(names[i], float(probs[i])) for i in topk_idx]


def main():
    st.set_page_config(page_title="Cattle Breed Classification", page_icon="🐄", layout="centered")
    st.title("🐄 Cattle Breed Classification")
    st.write("Upload an image to classify the cattle breed using a YOLOv8 classification model.")

    with st.sidebar:
        st.header("Settings")
        default_model = "yolov8_cattle_best.pt"
        model_path = st.text_input("Model path", value=default_model)
        confidence_threshold = st.slider("Confidence display threshold", 0.0, 1.0, 0.0, 0.01)

        model_load_placeholder = st.empty()
        try:
            with model_load_placeholder, st.spinner("Loading model..."):
                model = load_model(model_path)
            st.success("Model loaded")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.stop()

        cols = st.columns(2)
        with cols[0]:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running inference..."):
            results = model.predict(source=image, verbose=False)

        result = results[0]
        names = model.names

        if hasattr(result, "probs") and result.probs is not None:
            top1_idx = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf.item())
            top1_label = names[top1_idx]

            with cols[1]:
                st.subheader("Prediction")
                st.metric(label="Top-1 Breed", value=top1_label, delta=f"{top1_conf:.2%}")

                topk = get_topk_probs(result, names, k=min(5, len(names)))
                if topk:
                    st.write("Top probabilities:")
                    for label, prob in topk:
                        if prob < confidence_threshold:
                            continue
                        st.progress(prob, text=f"{label} — {prob:.2%}")
        else:
            st.warning("Model did not return classification probabilities. Ensure this is a classification model.")


if __name__ == "__main__":
    main()
