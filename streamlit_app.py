
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Brain Tumor Virtual Lab",
    page_icon="🧠",
    layout="wide"
)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "brain_tumor_vgg16.keras"
RESULTS_PATH = BASE / "brain_tumor_results"

CLASS_NAMES = ["Glioma", "Meningioma"]
IMG_SIZE = 224

st.title("🧠 Brain Tumor Virtual Lab")
st.caption("VGG16 Classification • LIME Explainable AI")

st.warning(
    "Educational prototype only. This website is not a medical "
    "diagnostic tool. Never use its predictions to make medical decisions."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

# ---------------- PREDICTION ----------------
def predict_batch(images):
    images = np.asarray(images, dtype=np.float32)

    # This model includes VGG16 preprocessing internally.
    # Do not preprocess the images again here.
    meningioma_scores = load_model().predict(
        images, verbose=0
    ).reshape(-1)

    return np.column_stack([
        1.0 - meningioma_scores,
        meningioma_scores
    ])

# ---------------- NAVIGATION ----------------
prediction_tab, performance_tab, learning_tab = st.tabs([
    "🔍 Predict & Explain",
    "📊 Model Performance",
    "📘 How It Works"
])

with prediction_tab:
    st.header("Upload a brain MRI scan")
    st.write(
        "Upload an image to see the model's prediction and a LIME "
        "visual explanation."
    )

    uploaded_image = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.asarray(image, dtype=np.float32)

        st.image(image, caption="Uploaded MRI scan", width=350)

        if st.button("Analyze MRI", type="primary"):
            with st.spinner("Analyzing image and generating LIME..."):
                try:
                    probabilities = predict_batch(
                        np.expand_dims(image_array, axis=0)
                    )[0]

                    predicted_index = int(np.argmax(probabilities))
                    predicted_class = CLASS_NAMES[predicted_index]

                    st.subheader(f"Prediction: {predicted_class}")
                    st.write(
                        "These are model scores, not clinically "
                        "validated probabilities."
                    )

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Glioma model score",
                        f"{probabilities[0] * 100:.2f}%"
                    )
                    col2.metric(
                        "Meningioma model score",
                        f"{probabilities[1] * 100:.2f}%"
                    )

                    st.progress(float(probabilities[predicted_index]))

                    explainer = lime_image.LimeImageExplainer()
                    explanation = explainer.explain_instance(
                        image_array.astype(np.double),
                        predict_batch,
                        labels=(predicted_index,),
                        hide_color=0,
                        num_samples=500
                    )

                    lime_image_array, mask = (
                        explanation.get_image_and_mask(
                            predicted_index,
                            positive_only=False,
                            num_features=8,
                            hide_rest=False
                        )
                    )

                    lime_image_array = np.clip(
                        lime_image_array, 0, 255
                    ).astype(np.uint8)

                    overlay = mark_boundaries(
                        lime_image_array / 255.0, mask
                    )

                    st.subheader("LIME explanation")
                    st.image(
                        overlay,
                        caption=(
                            "Highlighted regions show image areas "
                            "that influenced this model's prediction."
                        ),
                        width=450
                    )
                    st.info(
                        "LIME explains this model's behavior; it does "
                        "not confirm a tumor or identify a clinically "
                        "reliable tumor boundary."
                    )

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

with performance_tab:
    st.header("Model Training & Performance")

    metrics_file = RESULTS_PATH / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r") as file:
            metrics = json.load(file)

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Test accuracy",
            f"{metrics['test_accuracy'] * 100:.2f}%"
        )
        col2.metric(
            "Epochs completed",
            str(metrics["epochs_completed"])
        )

        if "test_roc_auc" in metrics:
            col3.metric(
                "Test ROC-AUC",
                f"{metrics['test_roc_auc']:.3f}"
            )

        report = metrics.get("classification_report", {})
        metric_cols = st.columns(2)

        for index, class_name in enumerate(
            ["glioma", "meningioma"]
        ):
            if class_name in report:
                with metric_cols[index]:
                    st.subheader(class_name.title())
                    st.write(
                        f"Precision: {report[class_name]['precision']:.3f}"
                    )
                    st.write(
                        f"Recall: {report[class_name]['recall']:.3f}"
                    )
                    st.write(
                        f"F1-score: {report[class_name]['f1-score']:.3f}"
                    )

    for filename, title in [
        ("accuracy.png", "Training and Validation Accuracy"),
        ("loss.png", "Training and Validation Loss"),
        ("confusion_matrix.png", "Independent Test Confusion Matrix")
    ]:
        image_path = RESULTS_PATH / filename
        st.subheader(title)

        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.info(f"Upload {filename} to display this graph.")

    st.caption(
        "Test results reflect the dataset and evaluation procedure "
        "used during training. They do not guarantee real-world accuracy."
    )

with learning_tab:
    st.header("How the model works")

    st.subheader("1. Dataset")
    st.write(
        "The project uses brain MRI images from the selected Glioma "
        "and Meningioma classes. Images are resized to 224 × 224 pixels."
    )

    st.subheader("2. VGG16 transfer learning")
    st.write(
        "VGG16 is a convolutional neural network pretrained on ImageNet. "
        "The pretrained feature extractor is adapted to distinguish "
        "the two selected MRI classes."
    )
    st.write(
        "The model uses a 256-unit ReLU dense layer, dropout, and a "
        "sigmoid output. Training first freezes the VGG16 feature "
        "extractor, then fine-tunes its final block."
    )

    st.subheader("3. Training")
    st.write(
        "The notebook uses training and validation datasets, class "
        "weights, early stopping, and learning-rate reduction. "
        "The maximum training schedule is 25 epochs."
    )

    st.subheader("4. Evaluation")
    st.write(
        "Accuracy, loss, a test-set confusion matrix, precision, recall, "
        "F1-score, and ROC-AUC are used to describe model performance."
    )

    st.subheader("5. Explainable AI with LIME")
    st.write(
        "LIME perturbs parts of an input image and observes how the "
        "model's outputs change. It builds a local explanation of "
        "which regions influenced an individual prediction."
    )

st.divider()
st.caption(
    "Brain Tumor Virtual Lab | VGG16 + LIME | "
    "Educational use only, not for clinical diagnosis."
)
