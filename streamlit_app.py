
import os
import json
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries, slic


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Bloom | AI Virtual Lab",
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
# 2. COLOURFUL AESTHETIC DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #FFF7FB 0%, #F4F0FF 48%, #EFFFF9 100%);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    color: #493B69 !important;
    font-family: 'Nunito', sans-serif !important;
}

.main-title {
    text-align: center;
    font-family: 'Nunito', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #7455A6;
    padding-top: 12px;
}

.subtitle {
    text-align: center;
    color: #88799D;
    font-size: 1.05rem;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #E9DFFF, #FFE4F0, #DDF8EF);
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    border: 1px solid #FFFFFF;
    margin-bottom: 20px;
}

.soft-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid #E8DDF6;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 5px 18px rgba(110, 78, 145, 0.06);
}

.disclaimer {
    background: #FFF1DC;
    border: 1px solid #F3D6A6;
    padding: 15px;
    border-radius: 14px;
    color: #77552C;
}

div.stButton > button {
    border-radius: 14px;
    border: 1px solid #DDC9F3;
    background: #FFFFFF;
    color: #624B86;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #F1E7FF;
    border-color: #B69ADE;
    color: #49336B;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #9675D4, #D78CB5);
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.85);
    padding: 15px;
    border: 1px solid #E9DFF5;
    border-radius: 16px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F1E9FF, #FFF0F6, #E9FFF6);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 10px 16px;
}

hr {
    border-color: #E6D9F1;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL DOWNLOAD AND LOADING
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        downloaded = gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False
        )

        if (
            not downloaded
            or not os.path.exists(MODEL_PATH)
            or os.path.getsize(MODEL_PATH) == 0
        ):
            raise RuntimeError(
                "Model download failed. Check the Google Drive "
                "file ID and sharing permissions."
            )

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
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )


try:
    with st.spinner("🌸 Preparing your AI laboratory..."):
        model = load_model()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.code(str(error))
    st.stop()


# =========================================================
# 4. IMAGE PREPARATION AND PREDICTION
# =========================================================

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32)


def predict_probabilities(images):
    batch = np.asarray(images, dtype=np.float32)

    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]

    predictions = np.asarray(model.predict(batch, verbose=0))

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape[-1] == 1:
        positive = predictions[:, 0]
        probabilities = np.column_stack(
            [1.0 - positive, positive]
        )
    elif predictions.shape[-1] == 2:
        probabilities = predictions
    else:
        raise ValueError(
            f"Expected 1 or 2 model outputs; "
            f"received {predictions.shape[-1]}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid probabilities.")

    # Do not silently reinterpret logits as probabilities.
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Model outputs are outside [0, 1]. Check whether "
            "the model returns logits instead of probabilities."
        )

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=0.02):
        raise ValueError(
            "Model outputs do not sum to approximately 1. "
            "Check the trained model's output layer."
        )

    return probabilities


# =========================================================
# 5. LIME EXPLANATION AND DIAGNOSTICS
# =========================================================

def make_lime_explanation(image_array, predicted_index, samples, seed):
    explainer = LimeImageExplainer(random_state=seed)

    explanation = explainer.explain_instance(
        image=image_array.astype(np.uint8),
        classifier_fn=predict_probabilities,
        labels=(predicted_index,),
        hide_color=0,
        num_samples=samples,
        segmentation_fn=lambda img: slic(
            img,
            n_segments=80,
            compactness=10,
            sigma=1,
            start_label=0,
            channel_axis=-1
        )
    )

    return explanation


def render_lime_overlay(image_array, explanation, label, num_features):
    segments = explanation.segments
    local_weights = dict(explanation.local_exp[label])

    ranked = sorted(
        local_weights.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:num_features]

    positive_segments = {
        segment_id for segment_id, weight in ranked if weight > 0
    }
    negative_segments = {
        segment_id for segment_id, weight in ranked if weight < 0
    }

    positive_mask = np.isin(
        segments, list(positive_segments)
    )
    negative_mask = np.isin(
        segments, list(negative_segments)
    )

    base = image_array.astype(np.float32) / 255.0
    overlay = base.copy()

    # Positive contributors: mint green.
    overlay[positive_mask] = (
        0.45 * overlay[positive_mask]
        + 0.55 * np.array([0.15, 0.85, 0.55])
    )

    # Negative contributors: coral red.
    overlay[negative_mask] = (
        0.45 * overlay[negative_mask]
        + 0.55 * np.array([1.0, 0.25, 0.35])
    )

    # Draw boundaries only around selected regions.
    selected_mask = positive_mask | negative_mask
    visual = mark_boundaries(
        overlay,
        segments,
        color=(1.0, 0.85, 0.15),
        mode="outer"
    )

    # Keep the explanation image legible.
    visual[~selected_mask] = (
        0.75 * visual[~selected_mask]
        + 0.25 * base[~selected_mask]
    )

    return np.clip(visual, 0, 1), local_weights, segments


def masking_test(image_array, segments, local_weights, label):
    ranked = sorted(
        local_weights.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    selected_ids = [segment_id for segment_id, _ in ranked[:5]]

    original_probability = float(
        predict_probabilities(image_array)[0, label]
    )

    masked = image_array.copy()

    # Replace selected regions with the image's mean pixel value.
    fill_value = image_array.mean(axis=(0, 1))

    region_mask = np.isin(segments, selected_ids)
    masked[region_mask] = fill_value

    masked_probability = float(
        predict_probabilities(masked)[0, label]
    )

    return original_probability, masked_probability


def stability_test(image_array, predicted_index, samples, first_weights):
    second = make_lime_explanation(
        image_array,
        predicted_index,
        samples,
        seed=2027
    )

    first_top = {
        segment_id
        for segment_id, _ in sorted(
            first_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:10]
    }

    second_weights = dict(second.local_exp[predicted_index])

    second_top = {
        segment_id
        for segment_id, _ in sorted(
            second_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:10]
    }

    union = first_top | second_top
    overlap = (
        len(first_top & second_top) / len(union)
        if union else 0.0
    )

    return overlap


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("# 🧠 Brain Bloom")
    st.caption("Your colourful AI learning lab")
    st.divider()

    st.markdown("### 🪻 Lab information")
    st.write("**Architecture:** VGG16")
    st.write("**Technique:** Transfer learning")
    st.write("**Explainer:** LIME")
    st.write("**Image size:** 224 × 224")
    st.write("**Classes:** Glioma, Meningioma")

    st.divider()

    st.markdown("### 🌷 Explore")
    st.write("🔬 MRI prediction")
    st.write("🧩 LIME explanations")
    st.write("📊 Model performance")
    st.write("🫀 Anatomy garden")

    st.divider()
    st.caption("Educational use only. Not a diagnostic tool.")


# =========================================================
# 7. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Bloom</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'A pastel-powered virtual lab for AI, MRI and explainability'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h2>Welcome to your little science universe 🌸</h2>
    <p>Explore brain MRI classification, discover model behaviour,
    and learn how explainable AI works.</p>
    <p>🧠 &nbsp; 🔬 &nbsp; 🧬 &nbsp; 🫀 &nbsp; 🌷</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<b>Important:</b> This project is an educational demonstration.
Predictions and LIME explanations can be incorrect. Highlighted
regions are not verified tumor boundaries and must not be used
to make medical decisions.
</div>
""", unsafe_allow_html=True)

st.write("")

tab_predict, tab_anatomy, tab_performance, tab_about = st.tabs([
    "🔬 MRI Lab",
    "🫀 Anatomy Garden",
    "📊 Model Results",
    "📘 About"
])


# =========================================================
# 8. MRI PREDICTION AND LIME
# =========================================================

with tab_predict:
    st.subheader("🌷 Upload your MRI")

    st.write(
        "Upload a JPG, JPEG or PNG image to explore the "
        "classifier's output."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"],
        key="mri_upload"
    )

    if uploaded_file is not None:
        try:
            original = Image.open(uploaded_file).convert("RGB")
            input_image = prepare_image(original)

            image_col, result_col = st.columns(2)

            with image_col:
                st.markdown("#### 🖼️ Your image")
                st.image(original, use_container_width=True)

            with result_col:
                st.markdown("#### 🪻 Model prediction")

                with st.spinner("Analysing the image..."):
                    probabilities = predict_probabilities(input_image)[0]

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.metric("Predicted class", predicted_class)
                st.metric("Model output score", f"{confidence * 100:.2f}%")

                for index, class_name in enumerate(CLASS_NAMES):
                    score = float(probabilities[index])
                    st.write(f"**{class_name}:** {score * 100:.2f}%")
                    st.progress(float(np.clip(score, 0, 1)))

                st.caption(
                    "This is a model score, not a calibrated "
                    "probability of a clinical diagnosis."
                )

            st.divider()
            st.subheader("🧩 Explain this prediction")

            samples = st.select_slider(
                "Perturbation samples",
                options=[500, 1000, 2000, 3000],
                value=2000,
                help="More samples take longer and do not guarantee validity."
            )

            num_features = st.slider(
                "Number of influential regions to display",
                min_value=3,
                max_value=15,
                value=8
            )

            if st.button(
                "✨ Generate LIME explanation",
                type="primary",
                use_container_width=True
            ):
                with st.spinner(
                    "Generating the explanation and running diagnostics..."
                ):
                    explanation = make_lime_explanation(
                        input_image,
                        predicted_index,
                        samples,
                        seed=42
                    )

                    visual, weights, segments = render_lime_overlay(
                        input_image.astype(np.uint8),
                        explanation,
                        predicted_index,
                        num_features
                    )

                    original_probability, masked_probability = masking_test(
                        input_image,
                        segments,
                        weights,
                        predicted_index
                    )

                    surrogate_score = getattr(
                        explanation, "score", None
                    )

                    st.session_state["lime_result"] = {
                        "visual": visual,
                        "weights": weights,
                        "segments": segments,
                        "score": surrogate_score,
                        "original_probability": original_probability,
                        "masked_probability": masked_probability,
                        "predicted_index": predicted_index,
                        "samples": samples,
                        "input_image": input_image.astype(np.uint8)
                    }

            result = st.session_state.get("lime_result")

            if result is not None:
                st.markdown("### 🌈 Your explanation")

                lime_col1, lime_col2 = st.columns(2)

                with lime_col1:
                    st.markdown("**Original resized MRI**")
                    st.image(
                        result["input_image"],
                        use_container_width=True
                    )

                with lime_col2:
                    st.markdown("**LIME region overlay**")
                    st.image(
                        result["visual"],
                        use_container_width=True
                    )

                st.markdown("""
                🟢 **Green:** regions supporting the selected class  
                🔴 **Red:** regions opposing the selected class  
                🟡 **Yellow:** superpixel boundaries
                """)

                st.caption(
                    "Colours indicate the fitted local explanation, "
                    "not confirmed tumor tissue."
                )

                st.divider()
                st.markdown("### 🔎 Explanation quality checks")

                score = result["score"]

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    if score is not None and np.isfinite(score):
                        st.metric("Local surrogate R²", f"{score:.3f}")
                    else:
                        st.metric("Local surrogate R²", "Unavailable")

                with metric_col2:
                    st.metric(
                        "Original class score",
                        f"{result['original_probability'] * 100:.2f}%"
                    )

                with metric_col3:
                    st.metric(
                        "Score after masking",
                        f"{result['masked_probability'] * 100:.2f}%"
                    )

                delta = (
                    result["masked_probability"]
                    - result["original_probability"]
                )

                st.write(
                    f"**Change after masking:** {delta * 100:+.2f} "
                    "percentage points."
                )

                st.caption(
                    "Masking can affect the prediction in either direction. "
                    "This single test is not sufficient to establish "
                    "explanation faithfulness."
                )

                st.markdown("#### 🌱 Repeatability check")

                if st.button("Check explanation stability"):
                    with st.spinner("Computing a second explanation..."):
                        overlap = stability_test(
                            result["input_image"].astype(np.float32),
                            result["predicted_index"],
                            result["samples"],
                            result["weights"]
                        )

                    st.metric(
                        "Top-region Jaccard overlap",
                        f"{overlap:.2f}"
                    )

                    if overlap >= 0.7:
                        st.success(
                            "The top-region overlap is relatively high "
                            "for this pair of runs."
                        )
                    else:
                        st.warning(
                            "The explanation changed noticeably. "
                            "Consider this result unstable."
                        )

                st.info(
                    "These checks are preliminary diagnostics, not a "
                    "complete validation protocol. R², masking sensitivity "
                    "and stability must be interpreted together."
                )

        except Exception as error:
            st.error("Unable to process the image or generate LIME.")
            with st.expander("Technical details"):
                st.exception(error)

    else:
        st.info("🌷 Upload an MRI image to begin.")


# =========================================================
# 9. CLICKABLE ANATOMY GARDEN
# =========================================================

with tab_anatomy:
    st.subheader("🫀 Welcome to the Anatomy Garden")

    st.write(
        "Click an organ card to discover a short educational fact. "
        "These are decorative learning cards, not medical assessments."
    )

    organs = {
        "🧠 Brain": (
            "The brain processes information, coordinates movement, "
            "and supports memory, learning and many other functions."
        ),
        "❤️ Heart": (
            "The heart pumps blood through the circulatory system, "
            "delivering oxygen and nutrients throughout the body."
        ),
        "🫁 Lungs": (
            "The lungs exchange oxygen and carbon dioxide during breathing."
        ),
        "🫘 Kidneys": (
            "The kidneys filter blood, regulate fluid balance, "
            "and help maintain electrolyte levels."
        ),
        "🧬 DNA": (
            "DNA stores hereditary information used in the development "
            "and functioning of living organisms."
        ),
        "🦴 Skeleton": (
            "The skeleton provides structural support, protects organs "
            "and works with muscles to enable movement."
        ),
        "🫀 Liver": (
            "The liver processes nutrients, produces bile and "
            "helps metabolise many substances."
        ),
        "🦷 Teeth": (
            "Teeth help break down food mechanically during digestion."
        )
    }

    if "selected_organ" not in st.session_state:
        st.session_state["selected_organ"] = "🧠 Brain"

    organ_items = list(organs.items())

    for start in range(0, len(organ_items), 4):
        cols = st.columns(4)

        for col, (organ, fact) in zip(
            cols, organ_items[start:start + 4]
        ):
            with col:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.markdown(f"### {organ}")
                if st.button(
                    "Explore ✨",
                    key=f"organ_{organ}",
                    use_container_width=True
                ):
                    st.session_state["selected_organ"] = organ
                st.markdown("</div>", unsafe_allow_html=True)

    selected = st.session_state["selected_organ"]

    st.markdown(
        f'<div class="hero"><h2>{selected}</h2>'
        f'<p>{organs[selected]}</p></div>',
        unsafe_allow_html=True
    )


# =========================================================
# 10. MODEL PERFORMANCE
# =========================================================

with tab_performance:
    st.subheader("📊 Model evaluation")

    st.write(
        "These are the evaluation results saved in your project. "
        "Only actual files and recorded metrics are displayed."
    )

    accuracy_path = os.path.join(RESULTS_DIR, "accuracy.png")
    loss_path = os.path.join(RESULTS_DIR, "loss.png")
    confusion_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.markdown("#### Recorded metrics")

            if isinstance(metrics, dict) and metrics:
                items = list(metrics.items())

                for start in range(0, len(items), 3):
                    cols = st.columns(3)

                    for col, (name, value) in zip(
                        cols, items[start:start + 3]
                    ):
                        if isinstance(value, (int, float)):
                            if 0 <= value <= 1:
                                display = f"{value * 100:.2f}%"
                            else:
                                display = f"{value:.4g}"
                        else:
                            display = str(value)

                        with col:
                            st.metric(
                                str(name).replace("_", " ").title(),
                                display
                            )
            else:
                st.json(metrics)

        except Exception as error:
            st.warning("Could not read metrics.json.")
            st.code(str(error))
    else:
        st.info("metrics.json was not found.")

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.markdown("#### 📈 Accuracy curve")
        if os.path.exists(accuracy_path):
            st.image(accuracy_path, use_container_width=True)
        else:
            st.info("accuracy.png was not found.")

    with graph_col2:
        st.markdown("#### 📉 Loss curve")
        if os.path.exists(loss_path):
            st.image(loss_path, use_container_width=True)
        else:
            st.info("loss.png was not found.")

    st.markdown("#### 🧮 Confusion matrix")
    if os.path.exists(confusion_path):
        st.image(confusion_path, use_container_width=True)
    else:
        st.info("confusion_matrix.png was not found.")

    st.caption(
        "Model performance metrics do not establish clinical suitability."
    )


# =========================================================
# 11. ABOUT THE PROJECT
# =========================================================

with tab_about:
    st.subheader("📘 How Brain Bloom works")

    st.markdown("""
    <div class="soft-card">
    <h3>1. 🖼️ MRI input</h3>
    The uploaded image is converted to RGB and resized to 224 × 224 pixels.
    </div>

    <div class="soft-card">
    <h3>2. 🧠 VGG16 classification</h3>
    The trained model returns outputs for the configured classes:
    Glioma and Meningioma.
    </div>

    <div class="soft-card">
    <h3>3. 🌱 Transfer learning</h3>
    A pretrained convolutional neural network is adapted to the target task.
    </div>

    <div class="soft-card">
    <h3>4. 🧩 LIME explainability</h3>
    LIME perturbs image superpixels and fits a local surrogate model
    to approximate the classifier's behaviour near the input image.
    </div>

    <div class="soft-card">
    <h3>5. 🔎 Explanation diagnostics</h3>
    Surrogate fit, masking sensitivity and repeatability help assess
    the explanation, but none independently proves correctness.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "This application is configured for Glioma and Meningioma only. "
        "It does not identify every type of brain tumor and is not "
        "a medical diagnostic system."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    "<center>🌸 <b>Brain Bloom Virtual Lab</b> · "
    "VGG16 + LIME · Made for learning 🧠</center>",
    unsafe_allow_html=True
)

st.caption(
    "Educational use only. Predictions and explanations can be wrong. "
    "Always consult qualified medical professionals for medical advice."
)
