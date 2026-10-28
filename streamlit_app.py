
import os
import json

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Classification Virtual Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "brain_tumor_vgg16.keras"
FILE_ID = "1aoglVHVDHw7VhlxLwRPnIe9ebpRSbmDd"
RESULTS_DIR = "brain_tumor_results"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma"]


# =========================================================
# 2. STYLING
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    text-align: center;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 1.5rem;
}
.disclaimer {
    padding: 14px;
    border-radius: 10px;
    background-color: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.5);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DOWNLOAD AND LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Download the model only if it is not already available.
    if (
        not os.path.exists(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) == 0
    ):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        download_url = f"https://drive.google.com/uc?id={FILE_ID}"

        downloaded_file = gdown.download(
            url=download_url,
            output=MODEL_PATH,
            quiet=False
        )

        if (
            not downloaded_file
            or not os.path.exists(MODEL_PATH)
            or os.path.getsize(MODEL_PATH) == 0
        ):
            raise RuntimeError(
                "Model download failed. Check that Google Drive "
                "sharing is set to Anyone with the link."
            )

    # Register the custom VGG16 preprocessing function.
    custom_objects = {
        "preprocess_input": preprocess_input,
        "keras.applications.vgg16.preprocess_input": preprocess_input,
    }

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects
        )

    except TypeError:
        # Compatibility fallback for TensorFlow/Keras versions
        # that do not accept the safe_mode parameter.
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )


try:
    with st.spinner("Loading the trained VGG16 model..."):
        model = load_model()

except Exception as error:
    st.error("Unable to load the trained model.")
    st.write(
        "Check the Google Drive link, model download, "
        "and TensorFlow/Keras compatibility."
    )

    with st.expander("Technical error details", expanded=True):
        st.code(str(error))

    st.stop()


# =========================================================
# 4. IMAGE PREPARATION AND PREDICTION
# =========================================================

def prepare_image(image):
    """Convert image to RGB and resize it to 224 x 224."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    return np.asarray(image, dtype=np.float32)


def predict_probabilities(images):

    batch = np.asarray(images, dtype=np.float32)

    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    # The model already contains its preprocessing Lambda layer.
    # Do not apply preprocess_input a second time.
    predictions = np.asarray(
        model.predict(batch, verbose=0)
    )

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # One sigmoid output.
    if predictions.shape[-1] == 1:
        positive = predictions[:, 0]

        probabilities = np.column_stack([
            1.0 - positive,
            positive
        ])

    # Two softmax outputs.
    elif predictions.shape[-1] == 2:
        probabilities = predictions

    else:
        raise ValueError(
            "Expected a binary model with one or two outputs. "
            f"Received {predictions.shape[-1]} outputs."
        )

    return probabilities


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🧠 Brain Tumor Virtual Lab")
    st.caption("VGG16 + LIME")

    st.divider()

    st.markdown("### Project Modules")
    st.markdown("""
    - MRI Image Prediction
    - LIME Explainability
    - Model Performance
    - About the Project
    """)

    st.divider()

    st.markdown("### Model Details")
    st.write("**Architecture:** VGG16")
    st.write("**Method:** Transfer learning")
    st.write("**Explanation:** LIME")
    st.write("**Input size:** 224 × 224")
    st.write("**Classes:** Glioma, Meningioma")


# =========================================================
# 6. HEADER AND DISCLAIMER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🧠 Brain Tumor Classification Virtual Lab'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore image classification using VGG16 and LIME'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer">
<b>Educational use only:</b> This project is a learning
demonstration, not a medical diagnostic tool. Model predictions
may be incorrect and must not be used to make medical decisions.
Consult a qualified medical professional for medical interpretation.
</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================================
# 7. APPLICATION TABS
# =========================================================

tab_predict, tab_performance, tab_about = st.tabs([
    "🔬 Predict & Explain",
    "📊 Model Performance",
    "📘 How It Works"
])


# =========================================================
# 8. PREDICT AND EXPLAIN
# =========================================================

with tab_predict:

    st.subheader("Upload an MRI Image")

    st.write(
        "Upload a JPG, JPEG, or PNG image to explore the "
        "trained binary classifier's output."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

            input_image = prepare_image(original_image)

            image_col, prediction_col = st.columns(2)

            with image_col:
                st.markdown("#### Uploaded Image")

                st.image(
                    original_image,
                    use_container_width=True
                )

            with prediction_col:
                st.markdown("#### Model Prediction")

                with st.spinner("Predicting..."):
                    probabilities = predict_probabilities(
                        input_image
                    )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(
                    probabilities[predicted_index]
                )

                st.metric(
                    "Predicted class",
                    predicted_class
                )

                st.metric(
                    "Model confidence score",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "The score reflects model output, not the "
                    "probability of a clinical diagnosis."
                )

                st.markdown("**Class scores**")

                for index, class_name in enumerate(CLASS_NAMES):
                    score = float(probabilities[index])

                    st.write(
                        f"{class_name}: {score * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0.0, 1.0))
                    )

            st.divider()

            st.subheader("LIME Explanation")

            st.write(
                "LIME highlights regions that influence the model's "
                "prediction for this particular image. These highlights "
                "are not verified tumor boundaries."
            )

            if st.button(
                "Generate LIME Explanation",
                type="primary"
            ):

                with st.spinner(
                    "Generating LIME explanation..."
                ):
                    explainer = LimeImageExplainer()

                    def lime_predict(images):
                        return predict_probabilities(images)

                    explanation = explainer.explain_instance(
                        input_image.astype(np.uint8),
                        classifier_fn=lime_predict,
                        labels=[predicted_index],
                        top_labels=None,
                        hide_color=0,
                        num_samples=500
                    )

                    lime_image, mask = (
                        explanation.get_image_and_mask(
                            label=predicted_index,
                            positive_only=False,
                            num_features=10,
                            hide_rest=False
                        )
                    )

                    visual = mark_boundaries(
                        lime_image / 255.0,
                        mask
                    )

                lime_col1, lime_col2 = st.columns(2)

                with lime_col1:
                    st.markdown("**Resized input**")

                    st.image(
                        input_image.astype(np.uint8),
                        use_container_width=True
                    )

                with lime_col2:
                    st.markdown("**LIME highlighted regions**")

                    st.image(
                        visual,
                        use_container_width=True
                    )

                st.caption(
                    "Green boundaries mark selected superpixels. "
                    "LIME explains model behaviour locally; it does "
                    "not establish clinical evidence."
                )

        except Exception as error:
            st.error("Unable to process this image or generate the explanation.")

            with st.expander("Technical error details"):
                st.code(str(error))

    else:
        st.info(
            "Upload an image to begin prediction and explanation."
        )


# =========================================================
# 9. MODEL PERFORMANCE
# =========================================================

with tab_performance:

    st.subheader("Model Evaluation Results")

    st.write(
        "These visualizations and metrics come from the saved "
        "evaluation files in the GitHub repository."
    )

    accuracy_path = os.path.join(
        RESULTS_DIR, "accuracy.png"
    )

    loss_path = os.path.join(
        RESULTS_DIR, "loss.png"
    )

    confusion_path = os.path.join(
        RESULTS_DIR, "confusion_matrix.png"
    )

    metrics_path = os.path.join(
        RESULTS_DIR, "metrics.json"
    )

    if os.path.exists(metrics_path):

        try:
            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as file:
                metrics = json.load(file)

            st.markdown("#### Evaluation Metrics")

            if isinstance(metrics, dict) and metrics:

                metric_items = list(metrics.items())
                columns = st.columns(
                    min(len(metric_items), 3)
                )

                for index, (name, value) in enumerate(metric_items):

                    if isinstance(value, (int, float)):
                        if 0 <= value <= 1:
                            display_value = (
                                f"{value * 100:.2f}%"
                            )
                        else:
                            display_value = f"{value:.4g}"
                    else:
                        display_value = str(value)

                    with columns[index % len(columns)]:
                        st.metric(
                            str(name).replace("_", " ").title(),
                            display_value
                        )

            else:
                st.json(metrics)

        except Exception as error:
            st.warning("Could not read metrics.json.")
            st.code(str(error))

    else:
        st.info(
            "metrics.json is missing from the "
            "brain_tumor_results folder."
        )

    st.divider()

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.markdown("#### Accuracy Curve")

        if os.path.exists(accuracy_path):
            st.image(
                accuracy_path,
                use_container_width=True
            )
        else:
            st.warning("accuracy.png was not found.")

    with graph_col2:
        st.markdown("#### Loss Curve")

        if os.path.exists(loss_path):
            st.image(
                loss_path,
                use_container_width=True
            )
        else:
            st.warning("loss.png was not found.")

    st.markdown("#### Confusion Matrix")

    if os.path.exists(confusion_path):
        st.image(
            confusion_path,
            use_container_width=True
        )
    else:
        st.warning("confusion_matrix.png was not found.")

    st.caption(
        "Evaluation results do not establish clinical suitability."
    )


# =========================================================
# 10. HOW IT WORKS
# =========================================================

with tab_about:

    st.subheader("About the Project")

    st.markdown("""
    #### 1. MRI Image Input
    The user uploads an MRI image. The application converts it to RGB
    and resizes it to 224 × 224 pixels.

    #### 2. VGG16 Classification
    The trained VGG16-based model produces scores for the two classes
    included in this project: glioma and meningioma.

    #### 3. Transfer Learning
    Transfer learning reuses features learned by a pretrained
    convolutional neural network and adapts them to a target task.

    #### 4. LIME Explainability
    LIME perturbs image regions and examines changes in model output
    to explain a prediction locally.

    #### 5. Performance Evaluation
    Accuracy and loss curves, the confusion matrix, and saved metrics
    summarize model evaluation.
    """)

    st.warning(
        "This model is configured for glioma and meningioma only. "
        "It is not a four-class classifier and does not identify "
        "pituitary tumors or no-tumor cases."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "Brain Tumor Classification Virtual Lab | VGG16 + LIME | "
    "Educational use only — not for medical diagnosis."
)
