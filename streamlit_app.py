
import os
import json
import hashlib
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
# 2. CHOCOLATE BROWN AND CREAM DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap');

:root {
    --chocolate: #4B2E24;
    --cocoa: #674438;
    --caramel: #B98255;
    --cream: #FFF9F1;
    --white: #FFFFFF;
    --muted: #705D53;
    --border: #E8D7C8;
}

.stApp {
    background: linear-gradient(
        135deg, #FFF9F1 0%, #F8F0E7 55%, #FDFBF7 100%
    );
    color: var(--chocolate);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] *,
.stRadio, .stCheckbox, label {
    color: var(--chocolate) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--chocolate) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
}

.main-title {
    text-align: center;
    font-family: 'Nunito', sans-serif;
    font-size: clamp(2.3rem, 5vw, 3.6rem);
    line-height: 1.15;
    font-weight: 800;
    color: #4B2E24;
    padding-top: 14px;
}

.subtitle {
    text-align: center;
    color: #705D53 !important;
    font-size: 1.08rem;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(
        120deg, #F2E1D1, #FFFDF9 58%, #EAD4C1
    );
    padding: 30px 26px;
    border-radius: 24px;
    text-align: center;
    border: 1px solid #E3CDBB;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(75,46,36,0.08);
}

.hero h2, .hero p {
    color: #4B2E24 !important;
}

.hero-art {
    font-size: clamp(3rem, 7vw, 5.2rem);
    line-height: 1.25;
    letter-spacing: .12em;
    margin: 12px 0;
}

.soft-card {
    background: rgba(255,255,255,0.96);
    color: #4B2E24;
    border: 1px solid #E8D7C8;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 7px 22px rgba(75,46,36,0.07);
}

.disclaimer {
    background: #FFF0D9;
    border: 1px solid #E7C89F;
    padding: 15px;
    border-radius: 14px;
    color: #603E25 !important;
}

div.stButton > button {
    border-radius: 14px;
    border: 1px solid #D9BDA7;
    background: #FFFFFF;
    color: #4B2E24 !important;
    font-weight: 800;
    min-height: 2.8rem;
    transition: all .2s ease;
}

div.stButton > button:hover {
    background: #F2E1D1;
    border-color: #B98255;
    transform: translateY(-1px);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #674438, #A56F4C);
    color: white !important;
    border: none;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.96);
    padding: 16px;
    border: 1px solid #E8D7C8;
    border-radius: 16px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #4B2E24 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F1E2D5, #FFF9F1);
    border-right: 1px solid #E8D7C8;
}

[data-testid="stSidebar"] * {
    color: #4B2E24 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.88);
    border-radius: 12px;
    padding: 10px 16px;
    color: #4B2E24 !important;
}

.stTabs [aria-selected="true"] {
    background: #EAD4C1 !important;
    color: #4B2E24 !important;
}

hr {
    border-color: #E8D7C8;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #B98255;
    border-radius: 16px;
    padding: 12px;
}

.stAlert {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL DOWNLOAD AND LOADING
# =========================================================

@st.cache_resource
def load_model():
    if (
        not os.path.exists(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) == 0
    ):
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
    with st.spinner("Preparing your AI laboratory..."):
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

    # Direct inference avoids the predict() wrapper overhead.
    predictions = model(batch, training=False)

    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()

    predictions = np.asarray(predictions)

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
# 5. LIME EXPLANATION
# =========================================================

def make_lime_explanation(image_array, predicted_index, samples, seed):
    explainer = LimeImageExplainer(random_state=seed)

    explanation = explainer.explain_instance(
        image=image_array.astype(np.uint8),
        classifier_fn=predict_probabilities,
        labels=(predicted_index,),
        hide_color=0,
        num_samples=samples,
        batch_size=64,
        segmentation_fn=lambda img: slic(
            img,
            n_segments=55,
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

    positive_mask = np.isin(segments, list(positive_segments))
    negative_mask = np.isin(segments, list(negative_segments))

    base = image_array.astype(np.float32) / 255.0
    overlay = base.copy()

    # Green indicates regions supporting the selected class.
    overlay[positive_mask] = (
        0.45 * overlay[positive_mask]
        + 0.55 * np.array([0.15, 0.85, 0.55])
    )

    # Red indicates regions opposing the selected class.
    overlay[negative_mask] = (
        0.45 * overlay[negative_mask]
        + 0.55 * np.array([1.0, 0.25, 0.35])
    )

    selected_mask = positive_mask | negative_mask

    visual = mark_boundaries(
        overlay,
        segments,
        color=(1.0, 0.85, 0.15),
        mode="outer"
    )

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

    return (
        len(first_top & second_top) / len(union)
        if union else 0.0
    )

# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:12px 0;">
            <div style="font-size:4rem;">🧠</div>
            <h2 style="margin:0;">Brain Bloom</h2>
            <p style="color:#705D53 !important;">
                Your AI learning laboratory
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧪 Lab information")
    st.write("**Architecture:** VGG16")
    st.write("**Technique:** Transfer learning")
    st.write("**Explainer:** LIME")
    st.write("**Image size:** 224 × 224")
    st.write("**Classes:** Glioma, Meningioma")

    st.divider()

    st.markdown("### 🧭 Explore your lab")
    st.write("🔬 MRI prediction")
    st.write("🧩 LIME explanations")
    st.write("📊 Model performance")
    st.write("🫀 Anatomy Garden")

    st.divider()

    st.caption(
        "🤎 Educational use only. Not a diagnostic tool."
    )


# =========================================================
# 7. HEADER AND HOMEPAGE
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Bloom</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        A smarter way to explore brain MRI classification
        and explainable artificial intelligence.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-art">🧠 🩺 🔬</div>
        <h2>Welcome to your AI medical learning lab!</h2>
        <p>
            Explore MRI image classification, understand model
            predictions and discover the regions highlighted by LIME.
        </p>
        <p>
            <b>Upload. Predict. Explain. Explore.</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Feature cards
feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.markdown(
        """
        <div class="soft-card" style="text-align:center;">
            <div style="font-size:3rem;">🧠</div>
            <h3>MRI Intelligence</h3>
            <p>Explore the model's classification of uploaded MRI images.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature2:
    st.markdown(
        """
        <div class="soft-card" style="text-align:center;">
            <div style="font-size:3rem;">🔎</div>
            <h3>Explainable AI</h3>
            <p>Explore image regions influencing the model's prediction.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature3:
    st.markdown(
        """
        <div class="soft-card" style="text-align:center;">
            <div style="font-size:3rem;">🩺</div>
            <h3>Learn Anatomy</h3>
            <p>Discover interactive educational facts about the human body.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="disclaimer">
        <b>Important medical notice:</b> This project is an
        educational demonstration. Model predictions and LIME
        explanations can be incorrect. Highlighted regions are not
        verified tumour boundaries. Do not use this application
        to diagnose or make medical decisions.
    </div>
    """,
    unsafe_allow_html=True
)

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
    st.subheader("🤎 Your MRI analysis lab")

    st.write(
        "Upload a JPG, JPEG or PNG image to explore "
        "the trained classifier's output."
    )

    uploaded_file = st.file_uploader(
        "📁 Choose your MRI image",
        type=["jpg", "jpeg", "png"],
        key="mri_upload"
    )

    # Identify the uploaded image so stale results are not reused.
    current_image_hash = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_image_hash = hashlib.sha256(file_bytes).hexdigest()

        if (
            st.session_state.get("current_image_hash")
            != current_image_hash
        ):
            st.session_state["current_image_hash"] = current_image_hash
            st.session_state.pop("lime_result", None)
            st.session_state.pop("lime_error", None)
            st.session_state.pop("stability_result", None)

        try:
            original = Image.open(uploaded_file).convert("RGB")
            input_image = prepare_image(original)

            image_col, result_col = st.columns([1.05, 1])

            with image_col:
                st.markdown("### 🖼️ Uploaded MRI")
                st.image(
                    original,
                    use_container_width=True
                )

                st.caption(
                    "The image is resized to 224 × 224 pixels "
                    "for model inference."
                )

            with result_col:
                st.markdown("### 🤎 Model prediction")

                prediction_key = f"prediction_{current_image_hash}"

                if prediction_key not in st.session_state:
                    with st.spinner("🔬 Analysing your MRI..."):
                        probabilities = predict_probabilities(
                            input_image
                        )[0]

                    st.session_state[prediction_key] = probabilities

                probabilities = np.asarray(
                    st.session_state[prediction_key]
                )

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.markdown(
                    f"""
                    <div class="soft-card">
                        <p style="margin:0;">PREDICTED CLASS</p>
                        <h2 style="font-size:2rem;margin-top:8px;">
                            {predicted_class}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Model output score",
                    f"{confidence * 100:.2f}%"
                )

                for index, class_name in enumerate(CLASS_NAMES):
                    score = float(probabilities[index])

                    st.write(
                        f"**{class_name}:** {score * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0, 1))
                    )

                st.caption(
                    "Model score is not a calibrated probability "
                    "of a clinical diagnosis."
                )

            st.divider()

            st.subheader("🧩 Explain this prediction")

            st.write(
                "Generate a LIME explanation to explore image "
                "regions that support or oppose the predicted class."
            )

            samples = st.select_slider(
                "Perturbation samples",
                options=[250, 500, 750, 1000, 2000],
                value=500,
                help=(
                    "Fewer samples are faster but may produce "
                    "less stable explanations. Compare settings "
                    "before selecting a final research value."
                )
            )

            num_features = st.slider(
                "Number of influential regions",
                min_value=3,
                max_value=15,
                value=8
            )

            generate_clicked = st.button(
                "✨ Generate LIME explanation",
                type="primary",
                use_container_width=True,
                key="generate_lime"
            )

            if generate_clicked:
                st.session_state.pop("lime_error", None)

                try:
                    progress = st.progress(
                        0,
                        text="Preparing image regions..."
                    )

                    with st.spinner(
                        "🧩 LIME is exploring the image. "
                        "Processing time depends on your hardware."
                    ):
                        progress.progress(
                            15,
                            text="Preparing image segments..."
                        )

                        explanation = make_lime_explanation(
                            input_image,
                            predicted_index,
                            samples,
                            seed=42
                        )

                        progress.progress(
                            80,
                            text="Creating the explanation overlay..."
                        )

                        visual, weights, segments = render_lime_overlay(
                            input_image.astype(np.uint8),
                            explanation,
                            predicted_index,
                            num_features
                        )

                        surrogate_score = getattr(
                            explanation, "score", None
                        )

                        # Save the main explanation before optional
                        # diagnostic calculations.
                        st.session_state["lime_result"] = {
                            "visual": visual,
                            "weights": weights,
                            "segments": segments,
                            "score": surrogate_score,
                            "predicted_index": predicted_index,
                            "samples": samples,
                            "input_image": input_image.astype(np.uint8),
                            "image_hash": current_image_hash,
                            "num_features": num_features
                        }

                        progress.progress(
                            100,
                            text="LIME explanation generated!"
                        )

                    st.session_state.pop("stability_result", None)

                except Exception as error:
                    st.session_state["lime_error"] = str(error)

            if st.session_state.get("lime_error"):
                st.error(
                    "Unable to generate the LIME explanation."
                )

                with st.expander("Technical details"):
                    st.code(st.session_state["lime_error"])

            result = st.session_state.get("lime_result")

            # Only show results belonging to the current upload.
            if (
                result is not None
                and result.get("image_hash") == current_image_hash
            ):
                st.divider()
                st.markdown("### 🌟 Your explanation")

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

                st.markdown(
                    """
                    <div class="soft-card">
                        <p>
                            🟢 <b>Green:</b> regions supporting the selected class
                        </p>
                        <p>
                            🔴 <b>Red:</b> regions opposing the selected class
                        </p>
                        <p>
                            🟡 <b>Yellow:</b> superpixel boundaries
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "Highlighted regions represent a local model "
                    "explanation, not confirmed tumour tissue."
                )

                st.divider()
                st.markdown("### 🔎 Explanation quality checks")

                score = result["score"]

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    if score is not None and np.isfinite(score):
                        st.metric(
                            "Local surrogate R²",
                            f"{score:.3f}"
                        )
                    else:
                        st.metric(
                            "Local surrogate R²",
                            "Unavailable"
                        )

                with metric_col2:
                    st.metric(
                        "Perturbation samples",
                        result["samples"]
                    )

                st.info(
                    "LIME's local surrogate score measures how well "
                    "its simplified explanation fits the model locally. "
                    "It does not establish clinical accuracy."
                )

                st.markdown("### 🧪 Optional diagnostic tests")

                with st.expander(
                    "Run masking sensitivity test"
                ):
                    st.write(
                        "This test masks the five most influential "
                        "regions and compares the selected class score."
                    )

                    if st.button(
                        "Run masking test",
                        key="run_masking"
                    ):
                        with st.spinner(
                            "Testing the selected image regions..."
                        ):
                            original_probability, masked_probability = (
                                masking_test(
                                    result["input_image"].astype(np.float32),
                                    result["segments"],
                                    result["weights"],
                                    result["predicted_index"]
                                )
                            )

                        st.metric(
                            "Original class score",
                            f"{original_probability * 100:.2f}%"
                        )

                        st.metric(
                            "Score after masking",
                            f"{masked_probability * 100:.2f}%"
                        )

                        delta = (
                            masked_probability - original_probability
                        )

                        st.write(
                            f"**Change after masking:** "
                            f"{delta * 100:+.2f} percentage points"
                        )

                        st.caption(
                            "Masking sensitivity is a preliminary "
                            "diagnostic and is not proof of explanation "
                            "faithfulness."
                        )

                with st.expander(
                    "Run explanation stability test"
                ):
                    st.write(
                        "Generate a second LIME explanation using a "
                        "different random seed and compare the top regions."
                    )

                    if st.button(
                        "Check explanation stability",
                        key="check_stability"
                    ):
                        with st.spinner(
                            "Computing a second explanation..."
                        ):
                            overlap = stability_test(
                                result["input_image"].astype(np.float32),
                                result["predicted_index"],
                                result["samples"],
                                result["weights"]
                            )

                        st.session_state["stability_result"] = {
                            "image_hash": current_image_hash,
                            "overlap": overlap
                        }

                    stability = st.session_state.get(
                        "stability_result"
                    )

                    if (
                        stability is not None
                        and stability.get("image_hash")
                        == current_image_hash
                    ):
                        overlap = stability["overlap"]

                        st.metric(
                            "Top-region Jaccard overlap",
                            f"{overlap:.2f}"
                        )

                        if overlap >= 0.7:
                            st.success(
                                "The top-region overlap is relatively "
                                "high for this pair of runs."
                            )
                        else:
                            st.warning(
                                "The explanation changed noticeably. "
                                "Consider this result unstable."
                            )

                st.caption(
                    "R², masking sensitivity and repeatability should "
                    "be interpreted together. None independently proves "
                    "that the highlighted regions are clinically correct."
                )

        except Exception as error:
            st.error(
                "Unable to process the uploaded image."
            )

            with st.expander("Technical details"):
                st.exception(error)

    else:
        st.markdown(
            """
            <div class="soft-card" style="text-align:center;">
                <div style="font-size:4rem;">🧠</div>
                <h3>Your MRI exploration starts here</h3>
                <p>
                    Upload an image above to view its model prediction
                    and generate an optional LIME explanation.
                </p>
                <div style="font-size:2rem;">🤎 🩺 🤎</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# 9. ANATOMY GARDEN
# =========================================================

with tab_anatomy:
    st.markdown("## 🫀 Anatomy Garden")
    st.write(
        "Explore a few educational facts about the brain "
        "and the systems that support it."
    )

    anatomy_items = {
        "🧠 The Brain": {
            "heading": "Your body's command centre",
            "facts": [
                "The brain contains billions of nerve cells.",
                "Different brain regions contribute to movement, "
                "memory, sensation and other functions.",
                "MRI sequences provide different kinds of information "
                "about brain structures and tissue."
            ],
            "icon": "🧠"
        },
        "🩸 Blood Vessels": {
            "heading": "The brain's supply network",
            "facts": [
                "Blood vessels deliver oxygen and nutrients to brain tissue.",
                "Blood flow is essential for normal brain function.",
                "Vascular structures may have different appearances "
                "across MRI sequences."
            ],
            "icon": "🩸"
        },
        "🦴 Skull": {
            "heading": "A protective structure",
            "facts": [
                "The skull protects the brain from many external impacts.",
                "MRI can depict soft tissues inside the skull.",
                "MRI and CT provide different types of diagnostic information."
            ],
            "icon": "🦴"
        },
        "🧬 Neurons": {
            "heading": "Cells that transmit information",
            "facts": [
                "Neurons communicate using electrical and chemical signals.",
                "Connections between neurons help support learning and memory.",
                "Routine MRI images do not directly show individual neurons."
            ],
            "icon": "🧬"
        }
    }

    selected_anatomy = st.selectbox(
        "Choose an anatomy topic",
        list(anatomy_items.keys()),
        key="anatomy_topic"
    )

    item = anatomy_items[selected_anatomy]

    st.markdown(
        f"""
        <div class="soft-card">
            <div style="font-size:4rem;text-align:center;">
                {item["icon"]}
            </div>
            <h2 style="text-align:center;">
                {item["heading"]}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    for number, fact in enumerate(item["facts"], start=1):
        st.markdown(
            f"""
            <div class="soft-card">
                <b>Did you know? {number}</b>
                <p>{fact}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "This section is educational. It does not identify or "
        "diagnose anatomical structures in your uploaded MRI."
    )


# =========================================================
# 10. MODEL RESULTS
# =========================================================

with tab_performance:
    st.markdown("## 📊 Model Results")
    st.write(
        "Review the model's output for your current MRI image "
        "and understand which evaluation measures are relevant "
        "to a brain tumour classification project."
    )

    st.markdown("### 🧠 Prediction score comparison")

    current_hash = st.session_state.get("current_image_hash")
    prediction_key = (
        f"prediction_{current_hash}" if current_hash else None
    )

    if (
        prediction_key
        and prediction_key in st.session_state
    ):
        scores = np.asarray(
            st.session_state[prediction_key]
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:
            st.metric(
                "Glioma model score",
                f"{float(scores[0]) * 100:.2f}%"
            )

        with score_col2:
            st.metric(
                "Meningioma model score",
                f"{float(scores[1]) * 100:.2f}%"
            )

        st.bar_chart(
            {
                "Model output (%)": {
                    "Glioma": float(scores[0]) * 100,
                    "Meningioma": float(scores[1]) * 100
                }
            },
            y_label="Score (%)"
        )

        st.caption(
            "These scores describe the current image only. "
            "They are not a dataset-level accuracy measurement."
        )
    else:
        st.info(
            "Upload an MRI in the MRI Lab tab to see its "
            "prediction scores here."
        )

    st.divider()

    st.markdown("### 📘 Evaluation measures for your research")

    evaluation_data = [
        {
            "Measure": "Accuracy",
            "Meaning": "Proportion of all predictions that are correct."
        },
        {
            "Measure": "Precision",
            "Meaning": "Proportion of predicted positive cases that are correct."
        },
        {
            "Measure": "Recall",
            "Meaning": "Proportion of actual positive cases identified."
        },
        {
            "Measure": "F1-score",
            "Meaning": "Harmonic mean of precision and recall."
        },
        {
            "Measure": "Confusion matrix",
            "Meaning": "Shows correct and incorrect predictions by class."
        }
    ]

    st.dataframe(
        evaluation_data,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Do not report a single uploaded image's score as model "
        "accuracy. Dataset-level metrics require labelled test data "
        "and predictions from the held-out test set."
    )

    st.markdown("### 🔬 VGG16 versus VGG16 + LIME")

    st.write(
        "LIME is an explanation technique applied to a model's "
        "predictions. Adding LIME does not automatically retrain "
        "VGG16 or improve its classification accuracy."
    )

    st.markdown(
        """
        <div class="soft-card">
            <h3>VGG16</h3>
            <p>
                Produces the classification output for an MRI image.
            </p>
            <hr>
            <h3>VGG16 + LIME</h3>
            <p>
                Uses the same classifier and adds an explanation of
                influential image regions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "For a fair research comparison, evaluate both approaches "
        "on the same held-out dataset. Report classification metrics "
        "separately from explanation quality and computational cost."
    )


# =========================================================
# 11. ABOUT PAGE
# =========================================================

with tab_about:
    st.markdown("## 📘 About Brain Bloom")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-art">🧠 🤎 🩺</div>
            <h2>Making AI easier to understand</h2>
            <p>
                Brain Bloom is an educational application for exploring
                brain MRI classification using VGG16 and explaining
                individual predictions with LIME.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎯 Project objectives")

    st.markdown(
        """
        - Explore brain MRI image classification.
        - Display the classifier's predicted class and output scores.
        - Visualize influential image regions using LIME.
        - Demonstrate optional masking and explanation stability tests.
        - Encourage transparent interpretation of AI outputs.
        """
    )

    st.markdown("### ⚙️ Technologies used")

    technology_col1, technology_col2 = st.columns(2)

    with technology_col1:
        st.markdown(
            """
            <div class="soft-card">
                <h3>🧠 Deep learning</h3>
                <p>TensorFlow and Keras</p>
                <p>VGG16 image classification</p>
                <p>NumPy image processing</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with technology_col2:
        st.markdown(
            """
            <div class="soft-card">
                <h3>🔎 Explainable AI</h3>
                <p>LIME image explanations</p>
                <p>Superpixel segmentation</p>
                <p>Streamlit user interface</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🩺 Responsible AI")

    st.write(
        "This application is a student research and educational "
        "demonstration. It has not been established as a clinically "
        "validated diagnostic system. Predictions may be wrong, "
        "and LIME explanations may be unstable or misleading."
    )

    st.markdown(
        """
        <div class="disclaimer">
            <b>For research and learning only.</b>
            Always rely on qualified healthcare professionals for
            medical interpretation and diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:18px 8px 30px 8px;
        color:#705D53;
    ">
        <div style="font-size:2.2rem;">🧠 🤎 🩺</div>
        <h3 style="margin-bottom:6px;">Brain Bloom</h3>
        <p>
            Explore the image. Understand the prediction.
            Keep humans at the centre of healthcare.
        </p>
        <p style="font-size:0.85rem;">
            Educational AI demonstration · VGG16 + LIME
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
