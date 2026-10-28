
import os
import json
import tempfile

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Virtual Lab",
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
# 2. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.35);
        margin-bottom: 12px;
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
# 3. MODEL DOWNLOAD AND LOADING
# =========================================================

@st.cache_resource(show_spinner="Downloading and loading the VGG16 model...")
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        downloaded_file = gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False
            
        )

        if (
            not downloaded_file
            or not os.path.exists(MODEL_PATH)
            or os.path.getsize(MODEL_PATH) == 0
        ):
            raise RuntimeError(
                "The model could not be downloaded. "
                "Check that Google Drive sharing is set to "
                "'Anyone with the link' and that the file is accessible."
            )

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
    except TypeError:
        # Compatibility fallback for older Keras versions.
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )


# =========================================================
# 4. PREDICTION FUNCTIONS
# =========================================================

def prepare_image(image):
    """Convert an image to RGB and resize it for VGG16."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32)


def predict_probabilities(images, model):
    """
    Predict probabilities for a batch of RGB images.

    The saved model contains its own VGG16 preprocessing layer.
    Therefore, input pixels remain in the 0-255 range.
    """
    batch = np.asarray(images, dtype=np.float32)

    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    predictions = np.asarray(
        model.predict(batch, verbose=0)
    )

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # Binary model with one sigmoid output.
    if predictions.shape[-1] == 1:
        positive_probability = predictions[:, 0]
        probabilities = np.column_stack([
            1.0 - positive_probability,
            positive_probability
        ])

    # Binary model with two softmax outputs.
    elif predictions.shape[-1] == 2:
        probabilities = predictions

    else:
        raise ValueError(
            f"Expected a binary model with 1 or 2 outputs, "
            f"but received {predictions.shape[-1]} outputs."
        )

    return probabilities


# =========================================================
# 5. LOAD MODEL
# =========================================================

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.info(
        "Check the Google Drive sharing permissions, model file ID, "
        "and the deployment logs."
    )
    with st.expander("Technical error details"):
        st.code(str(error))
    st.stop()


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🧠 Brain Tumor Virtual Lab")
    st.caption("VGG16 + LIME | Educational AI Project")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("- MRI Prediction")
    st.markdown("- LIME Explanation")
    st.markdown("- Model Performance")
    st.markdown("- How It Works")

    st.divider()

    st.markdown("### Model Information")
    st.write("**Architecture:** VGG16")
    st.write("**Learning method:** Transfer learning")
    st.write("**Explainability:** LIME")
    st.write("**Input:** MRI image")
    st.write("**Classes:** Glioma, Meningioma")


# =========================================================
# 7. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Classification Virtual Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore deep learning predictions and visual explanations using VGG16 and LIME'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer">
<b>Educational disclaimer:</b> This project is intended only for
learning and demonstration. It is not a medical device and must not
be used to diagnose, exclude, or treat a brain tumor. Model predictions
may be incorrect. Consult a qualified medical professional for medical
interpretation.
</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================================
# 8. TABS
# =========================================================

tab_predict, tab_performance, tab_about = st.tabs([
    "🔬 Predict & Explain",
    "📊 Model Performance",
    "📘 How It Works"
])


# =========================================================
# 9. PREDICTION AND LIME EXPLANATION
# =========================================================

with tab_predict:

    st.subheader("Upload an MRI Image")

    st.write(
        "Upload an image to explore how the trained binary classifier "
        "responds to it."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:

        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            input_image = prepare_image(original_image)

            left, right = st.columns([1, 1])

            with left:
                st.markdown("#### Uploaded Image")
                st.image(
                    original_image,
                    caption="Uploaded MRI image",
                    use_container_width=True
                )

            with right:
                st.markdown("#### Prediction")

                with st.spinner("Analysing image..."):
                    probabilities = predict_probabilities(
                        input_image,
                        model
                    )[0]

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.metric(
                    "Model prediction",
                    predicted_class
                )

                st.metric(
                    "Model confidence score",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "This score reflects the model's output, "
                    "not the probability of a clinical diagnosis."
                )

                st.markdown("**Output scores**")
                for index, class_name in enumerate(CLASS_NAMES):
                    st.write(
                        f"{class_name}: "
                        f"{probabilities[index] * 100:.2f}%"
                    )
                    st.progress(
                        float(np.clip(probabilities[index], 0.0, 1.0))
                    )

            st.divider()

            st.subheader("LIME: Explain the Prediction")

            st.write(
                "LIME highlights image regions that influence the model's "
                "prediction for this particular image. The highlighted "
                "areas are explanations of model behaviour, not confirmed "
                "tumor boundaries."
            )

            if st.button("Generate LIME Explanation", type="primary"):

                with st.spinner(
                    "Generating explanation. This may take some time..."
                ):
                    try:
                        explainer = LimeImageExplainer()

                        def lime_predict(batch):
                            return predict_probabilities(batch, model)

                        explanation = explainer.explain_instance(
                            input_image.astype(np.uint8),
                            classifier_fn=lime_predict,
                            labels=[predicted_index],
                            top_labels=None,
                            hide_color=0,
                            num_samples=500
                        )

                        lime_image, mask = explanation.get_image_and_mask(
                            label=predicted_index,
                            positive_only=False,
                            num_features=10,
                            hide_rest=False
                        )

                        highlighted_image = mark_boundaries(
                            lime_image / 255.0,
                            mask
                        )

                        st.success("LIME explanation generated.")

                        lime_col1, lime_col2 = st.columns(2)

                        with lime_col1:
                            st.markdown("**Original resized image**")
                            st.image(
                                input_image.astype(np.uint8),
                                use_container_width=True
                            )

                        with lime_col2:
                            st.markdown("**LIME highlighted regions**")
                            st.image(
                                highlighted_image,
                                use_container_width=True
                            )

                        st.caption(
                            "Green boundaries identify selected superpixels. "
                            "Positive and negative regions can both appear "
                            "because positive_only=False."
                        )

                    except Exception as lime_error:
                        st.error("Could not generate the LIME explanation.")
                        with st.expander("Technical error details"):
                            st.code(str(lime_error))

        except Exception as image_error:
            st.error("Could not process this image.")
            st.caption("Try another valid JPG or PNG image.")
            with st.expander("Technical error details"):
                st.code(str(image_error))

    else:
        st.info(
            "Upload an MRI image above to explore the model prediction "
            "and generate a LIME explanation."
        )


# =========================================================
# 10. MODEL PERFORMANCE
# =========================================================

with tab_performance:

    st.subheader("Model Evaluation Results")

    st.write(
        "The following results are loaded from the evaluation files "
        "generated by the training notebook."
    )

    accuracy_path = os.path.join(RESULTS_DIR, "accuracy.png")
    loss_path = os.path.join(RESULTS_DIR, "loss.png")
    confusion_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

    available_metrics = False

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.markdown("#### Evaluation Metrics")

            if isinstance(metrics, dict):
                metric_columns = st.columns(
                    min(max(len(metrics), 1), 3)
                )

                for index, (name, value) in enumerate(metrics.items()):
                    with metric_columns[index % len(metric_columns)]:
                        if isinstance(value, (int, float)):
                            if 0 <= value <= 1:
                                display_value = f"{value * 100:.2f}%"
                            else:
                                display_value = f"{value:.4g}"
                        else:
                            display_value = str(value)

                        st.metric(
                            str(name).replace("_", " ").title(),
                            display_value
                        )

                available_metrics = True
            else:
                st.json(metrics)
                available_metrics = True

        except Exception as metrics_error:
            st.warning("The metrics file could not be read.")
            st.caption(str(metrics_error))

    if not available_metrics:
        st.info(
            "Metrics are not available yet. Add metrics.json to the "
            "brain_tumor_results folder in your GitHub repository."
        )

    st.divider()

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.markdown("#### Accuracy Curve")

        if os.path.exists(accuracy_path):
            st.image(
                accuracy_path,
                caption="Training and validation accuracy",
                use_container_width=True
            )
        else:
            st.info("accuracy.png has not been added yet.")

    with graph_col2:
        st.markdown("#### Loss Curve")

        if os.path.exists(loss_path):
            st.image(
                loss_path,
                caption="Training and validation loss",
                use_container_width=True
            )
        else:
            st.info("loss.png has not been added yet.")

    st.markdown("#### Confusion Matrix")

    if os.path.exists(confusion_path):
        st.image(
            confusion_path,
            caption="Confusion matrix from model evaluation",
            use_container_width=True
        )
    else:
        st.info("confusion_matrix.png has not been added yet.")

    st.caption(
        "Performance results depend on the dataset and evaluation procedure. "
        "They do not establish clinical suitability."
    )


# =========================================================
# 11. HOW IT WORKS
# =========================================================

with tab_about:

    st.subheader("How the Virtual Lab Works")

    st.markdown("""
    #### 1. Image Input
    A user uploads an MRI image in JPG, JPEG, or PNG format.
    The image is converted to RGB and resized to 224 × 224 pixels.

    #### 2. VGG16 Classification
    The trained VGG16-based model processes the image and returns scores
    for the two classes used in this project: glioma and meningioma.

    #### 3. Transfer Learning
    VGG16 is a convolutional neural network architecture commonly used
    for image classification. Transfer learning reuses learned image
    features and adapts the model to a specific classification task.

    #### 4. LIME Explainability
    LIME perturbs regions of the uploaded image and observes how the
    model's output changes. It highlights regions that influence the
    selected prediction locally.

    #### 5. Performance Evaluation
    Accuracy and loss curves, the confusion matrix, and saved metrics
    help demonstrate how the model performed during evaluation.
    """)

    st.warning(
        "This demonstration covers only glioma and meningioma. "
        "It does not classify pituitary tumors or no-tumor images, "
        "even if such images are uploaded."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "Brain Tumor Classification Virtual Lab | VGG16 + LIME | "
    "Educational use only — not for medical diagnosis."
)
