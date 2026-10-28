import os
import json
import hashlib
import inspect

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries, slic

# ============================================================
# 1. CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Brain Bloom | AI Virtual Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "brain_tumor_vgg16.keras"
GOOGLE_DRIVE_FILE_ID = "1aoglVHVDHw7VhlxLwRPnIe9ebpRSbmDd"
RESULTS_DIR = "brain_tumor_results"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma"]

# Fixed LIME settings: no sliders in the website.
# 250 samples is a lighter starting point for Streamlit Cloud.
# Check explanation stability before using these settings in research.
LIME_SAMPLES = 250
LIME_FEATURES = 8
LIME_BATCH_SIZE = 16

# ============================================================
# 2. CHOCOLATE-BROWN + CREAM STYLING
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap');
    :root { --chocolate:#4B2E24; --cocoa:#674438; --caramel:#A87552;
            --cream:#FFF9F1; --border:#E8D7C8; --muted:#705D53; }
    .stApp { background:linear-gradient(135deg,#FFF9F1 0%,#F8F0E7 55%,#FDFBF7 100%);
             color:var(--chocolate); font-family:'DM Sans',sans-serif; }
    [data-testid="stAppViewContainer"],[data-testid="stHeader"] { background:transparent; }
    [data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"],
    [data-testid="stFileUploader"] *,label { color:var(--chocolate) !important; }
    h1,h2,h3,h4,h5,h6 { color:var(--chocolate) !important;
        font-family:'Nunito',sans-serif !important; font-weight:800 !important; }
    .main-title { text-align:center; font-family:'Nunito',sans-serif;
        font-size:clamp(2.4rem,5vw,3.6rem); line-height:1.15; font-weight:800;
        color:#4B2E24; padding-top:12px; }
    .subtitle { text-align:center; color:#705D53 !important; font-size:1.05rem; margin-bottom:25px; }
    .hero { background:linear-gradient(120deg,#F2E1D1,#FFFDF9 58%,#EAD4C1);
        padding:28px 24px; border-radius:24px; text-align:center; border:1px solid #E3CDBB;
        margin-bottom:20px; box-shadow:0 10px 30px rgba(75,46,36,.08); }
    .hero h2,.hero p { color:#4B2E24 !important; }
    .hero-art { font-size:clamp(3rem,7vw,5rem); line-height:1.3; margin:10px 0; }
    .soft-card { background:rgba(255,255,255,.97); color:#4B2E24; border:1px solid #E8D7C8;
        border-radius:20px; padding:20px; margin-bottom:12px;
        box-shadow:0 7px 22px rgba(75,46,36,.07); }
    .disclaimer { background:#FFF0D9; border:1px solid #E7C89F; padding:15px;
        border-radius:14px; color:#603E25 !important; }
    div.stButton > button { border-radius:14px; border:1px solid #D9BDA7;
        background:#FFF; color:#4B2E24 !important; font-weight:800; min-height:2.8rem;
        transition:all .2s ease; }
    div.stButton > button:hover { background:#F2E1D1; border-color:#B98255; transform:translateY(-1px); }
    div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#674438,#A56F4C);
        color:#FFF !important; border:none; }
    [data-testid="stMetric"] { background:rgba(255,255,255,.96); padding:15px;
        border:1px solid #E8D7C8; border-radius:16px; }
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"] {
        color:#4B2E24 !important; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#F1E2D5,#FFF9F1);
        border-right:1px solid #E8D7C8; }
    [data-testid="stSidebar"] * { color:#4B2E24 !important; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
    .stTabs [data-baseweb="tab"] { background:rgba(255,255,255,.88); border-radius:12px;
        padding:10px 16px; color:#4B2E24 !important; }
    .stTabs [aria-selected="true"] { background:#EAD4C1 !important; }
    [data-testid="stFileUploader"] { background:#FFF; border:1px dashed #B98255;
        border-radius:16px; padding:12px; }
    hr { border-color:#E8D7C8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        downloaded = gdown.download(url=url, output=MODEL_PATH, quiet=False)
        if not downloaded or not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            raise RuntimeError(
                "Model download failed. Check the Google Drive file ID and sharing permissions."
            )
    custom_objects = {"preprocess_input": tf.keras.applications.vgg16.preprocess_input}
    load_kwargs = {"compile": False, "custom_objects": custom_objects}
    # safe_mode is supported by newer Keras versions only.
    if "safe_mode" in inspect.signature(tf.keras.models.load_model).parameters:
        load_kwargs["safe_mode"] = False
    return tf.keras.models.load_model(MODEL_PATH, **load_kwargs)

try:
    with st.spinner("Preparing your AI laboratory..."):
        model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()

# Trace the model call once so repeated LIME batches avoid some eager-call overhead.
# The dynamic first dimension allows the final batch to contain fewer images.
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=tf.float32)
    ],
    reduce_retracing=True,
)
def run_model_inference(batch):
    return model(batch, training=False)

# ============================================================
# 4. IMAGE PREPARATION + CLASS PREDICTION
# ============================================================
def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def predict_probabilities(images):
    """Prediction callback used by both the main page and LIME."""
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]
    outputs = run_model_inference(tf.convert_to_tensor(batch, dtype=tf.float32))
    if hasattr(outputs, "numpy"):
        outputs = outputs.numpy()
    outputs = np.asarray(outputs)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if outputs.shape[-1] == 1:
        positive = outputs[:, 0]
        probs = np.column_stack((1.0 - positive, positive))
    elif outputs.shape[-1] == 2:
        probs = outputs
    else:
        raise ValueError(f"Expected 1 or 2 model outputs; received {outputs.shape[-1]}.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("The model returned non-finite values.")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(
            "Model outputs are outside [0, 1]. Check whether the model returns logits."
        )
    if not np.allclose(probs.sum(axis=1), 1.0, atol=0.02):
        raise ValueError(
            "Model outputs do not sum to approximately 1. Check the model output layer."
        )
    return probs

# ============================================================
# 5. LIME EXPLANATION
# ============================================================
def segment_image(image):
    """Support both newer and older scikit-image SLIC APIs."""
    try:
        return slic(
            image, n_segments=35, compactness=10, sigma=1,
            start_label=0, channel_axis=-1
        )
    except TypeError:
        # Compatibility with older scikit-image releases.
        return slic(
            image, n_segments=35, compactness=10, sigma=1,
            start_label=0, multichannel=True
        )


def make_lime_explanation(image_array, predicted_index, samples=LIME_SAMPLES, seed=42):
    explainer = LimeImageExplainer(random_state=seed)
    return explainer.explain_instance(
        image=image_array.astype(np.uint8),
        classifier_fn=predict_probabilities,
        labels=(int(predicted_index),),
        hide_color=0,
        num_samples=int(samples),
        batch_size=LIME_BATCH_SIZE,
        segmentation_fn=segment_image,
    )


def render_lime_overlay(image_array, explanation, label, num_features=LIME_FEATURES):
    """Render only influential superpixels that overlap the MRI foreground.

    This avoids highlighting large, nearly black background superpixels and
    avoids drawing boundaries around every segment in the image.
    """
    segments = explanation.segments
    weights = dict(explanation.local_exp[int(label)])

    # Exclude near-black scanner/background pixels from the display only.
    # This is a visualization filter, not a tumour/background segmentation.
    foreground = np.max(image_array, axis=2) > 12
    candidate_ids = []
    for segment_id in weights:
        region = (segments == segment_id)
        if np.any(region) and float(foreground[region].mean()) >= 0.50:
            candidate_ids.append(segment_id)

    ranked = sorted(
        ((sid, weights[sid]) for sid in candidate_ids),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:num_features]

    positive_ids = {sid for sid, weight in ranked if weight > 0}
    negative_ids = {sid for sid, weight in ranked if weight < 0}
    positive_mask = np.isin(segments, list(positive_ids)) & foreground
    negative_mask = np.isin(segments, list(negative_ids)) & foreground
    selected_mask = positive_mask | negative_mask

    base = image_array.astype(np.float32) / 255.0
    overlay = base.copy()
    overlay[positive_mask] = (
        0.45 * overlay[positive_mask]
        + 0.55 * np.array([0.15, 0.85, 0.55])
    )
    overlay[negative_mask] = (
        0.45 * overlay[negative_mask]
        + 0.55 * np.array([1.0, 0.25, 0.35])
    )

    # Keep boundaries only around the selected influential regions.
    selected_segments = np.full(segments.shape, -1, dtype=np.int32)
    selected_segments[selected_mask] = segments[selected_mask]
    visual = mark_boundaries(
        overlay, selected_segments, color=(1.0, 0.85, 0.15), mode="outer"
    )
    visual[~selected_mask] = (
        0.85 * visual[~selected_mask] + 0.15 * base[~selected_mask]
    )
    return np.clip(visual, 0, 1), weights, segments


def masking_test(image_array, segments, weights, label):
    ranked = sorted(weights.items(), key=lambda pair: abs(pair[1]), reverse=True)
    top_ids = [sid for sid, _ in ranked[:5]]
    original_score = float(predict_probabilities(image_array)[0, label])
    masked = image_array.copy()
    masked[np.isin(segments, top_ids)] = image_array.mean(axis=(0, 1))
    masked_score = float(predict_probabilities(masked)[0, label])
    return original_score, masked_score


def stability_test(image_array, predicted_index, samples, first_weights):
    second = make_lime_explanation(image_array, predicted_index, samples=samples, seed=2027)
    first_top = {sid for sid, _ in sorted(first_weights.items(), key=lambda pair: abs(pair[1]), reverse=True)[:10]}
    second_weights = dict(second.local_exp[int(predicted_index)])
    second_top = {sid for sid, _ in sorted(second_weights.items(), key=lambda pair: abs(pair[1]), reverse=True)[:10]}
    union = first_top | second_top
    return len(first_top & second_top) / len(union) if union else 0.0

# ============================================================
# 6. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:10px 0;">
        <div style="font-size:4rem;">🧠</div><h2 style="margin:0;">Brain Bloom</h2>
        <p>Your AI learning laboratory</p></div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 🧪 Lab information")
    st.write("**Architecture:** VGG16")
    st.write("**Explainer:** LIME")
    st.write("**Image size:** 224 × 224")
    st.write("**Classes:** Glioma, Meningioma")
    st.divider()
    st.markdown("### 🧭 Explore")
    st.write("🔬 MRI prediction")
    st.write("🧩 LIME explanation")
    st.write("📊 Model performance")
    st.write("🫀 Anatomy Garden")
    st.divider()
    st.caption("Educational use only. Not a diagnostic tool.")

# ============================================================
# 7. HOMEPAGE
# ============================================================
st.markdown('<div class="main-title">🧠 Brain Bloom</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A warmer way to explore brain MRI classification and explainable AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="hero"><div class="hero-art">🧠 🩺 🔬</div>
    <h2>Welcome to your AI medical learning lab!</h2>
    <p>Explore MRI classification, understand model predictions, and discover how LIME explains individual results.</p>
    <p><b>Upload. Predict. Explain. Explore.</b></p></div>""",
    unsafe_allow_html=True,
)

features = st.columns(3)
feature_cards = [
    ("🧠", "MRI Intelligence", "Explore the trained model output for an uploaded MRI."),
    ("🔎", "Explainable AI", "Visualize image regions that influence a prediction."),
    ("🩺", "Anatomy Garden", "Explore interactive facts about the human body."),
]
for col, (icon, title, description) in zip(features, feature_cards):
    with col:
        st.markdown(
            f'<div class="soft-card" style="text-align:center;"><div style="font-size:3rem;">{icon}</div><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
st.markdown(
    '<div class="disclaimer"><b>Important:</b> This is an educational demonstration. Predictions and LIME explanations can be incorrect. Highlighted regions are not verified tumour boundaries and must not be used for medical decisions.</div>',
    unsafe_allow_html=True,
)
st.write("")

tab_predict, tab_anatomy, tab_performance, tab_about = st.tabs(
    ["🔬 MRI Lab", "🫀 Anatomy Garden", "📊 Model Results", "📘 About"]
)

# ============================================================
# 8. MRI PREDICTION + LIME (NO SLIDERS)
# ============================================================
with tab_predict:
    st.markdown("## 🤎 Your MRI analysis lab")
    st.write("Upload an MRI to see the VGG16 classification and optionally generate a LIME explanation.")
    uploaded_file = st.file_uploader("📁 Choose your MRI image", type=["jpg", "jpeg", "png"], key="mri_upload")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        image_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("current_image_hash") != image_hash:
            st.session_state["current_image_hash"] = image_hash
            for key in ("lime_result", "lime_error", "masking_result", "stability_result", "prediction"):
                st.session_state.pop(key, None)

        try:
            original = Image.open(uploaded_file).convert("RGB")
            input_image = prepare_image(original)
            image_col, result_col = st.columns([1.05, 1])
            with image_col:
                st.markdown("### 🖼️ Uploaded MRI")
                st.image(original, use_container_width=True)
                st.caption("The model input is resized to 224 × 224 pixels.")
            with result_col:
                st.markdown("### 🩺 Model prediction")
                if "prediction" not in st.session_state:
                    with st.spinner("Analysing the MRI image..."):
                        st.session_state["prediction"] = predict_probabilities(input_image)[0]
                probabilities = np.asarray(st.session_state["prediction"])
                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])
                st.markdown(
                    f'<div class="soft-card" style="text-align:center;"><div style="font-size:3rem;">🧠</div><p>PREDICTED CLASS</p><h2 style="font-size:2rem;">{predicted_class}</h2></div>',
                    unsafe_allow_html=True,
                )
                st.metric("Model output score", f"{confidence * 100:.2f}%")
                st.markdown("#### Class scores")
                for index, class_name in enumerate(CLASS_NAMES):
                    score = float(probabilities[index])
                    st.write(f"**{class_name}: {score * 100:.2f}%**")
                    st.progress(float(np.clip(score, 0.0, 1.0)))
                st.caption("A model score is not a calibrated probability of clinical diagnosis.")

            st.divider()
            st.markdown("## 🧩 Explain this prediction")
            st.write(
                "LIME highlights image regions that influence the model's prediction."
            )
            generate_clicked = st.button(
                "✨ Generate LIME explanation", type="primary", use_container_width=True, key="generate_lime"
            )

            if generate_clicked:
                existing = st.session_state.get("lime_result")
                if existing is not None and existing.get("image_hash") == image_hash:
                    st.info("A LIME explanation is already available for this image.")
                else:
                    st.session_state.pop("lime_error", None)
                    try:
                        with st.spinner(
                            "Generating LIME explanation. The first run can take longer while TensorFlow prepares inference."
                        ):
                            explanation = make_lime_explanation(
                                input_image.astype(np.uint8),
                                predicted_index,
                                samples=LIME_SAMPLES,
                                seed=42,
                            )
                            visual, weights, segments = render_lime_overlay(
                                input_image.astype(np.uint8),
                                explanation,
                                predicted_index,
                                LIME_FEATURES,
                            )
                            score = getattr(explanation, "score", None)
                            st.session_state["lime_result"] = {
                                "visual": visual,
                                "weights": weights,
                                "segments": segments,
                                "score": score,
                                "predicted_index": predicted_index,
                                "samples": LIME_SAMPLES,
                                "input_image": input_image.astype(np.uint8),
                                "image_hash": image_hash,
                            }
                    except Exception as exc:
                        st.session_state["lime_error"] = f"{type(exc).__name__}: {exc}"

            if st.session_state.get("lime_error"):
                st.error("Unable to generate the LIME explanation.")
                with st.expander("Technical error details"):
                    st.code(st.session_state["lime_error"])

            result = st.session_state.get("lime_result")
            if result is not None and result.get("image_hash") == image_hash:
                st.divider()
                st.markdown("### 🌟 Your LIME explanation")
                lime_col1, lime_col2 = st.columns(2)
                with lime_col1:
                    st.markdown("**Original resized MRI**")
                    st.image(result["input_image"], use_container_width=True)
                with lime_col2:
                    st.markdown("**LIME region overlay**")
                    st.image(result["visual"], use_container_width=True)
                st.markdown(
                    """<div class="soft-card"><p>🟢 <b>Green:</b> regions supporting the selected class</p>
                    <p>🔴 <b>Red:</b> regions opposing the selected class</p>
                    <p>🟡 <b>Yellow:</b> superpixel boundaries</p></div>""",
                    unsafe_allow_html=True,
                )
                st.caption("Highlighted regions are local explanations, not confirmed tumour tissue.")
                score = result["score"]
                if score is not None and np.isfinite(score):
                    st.metric("LIME local surrogate score (R²)", f"{score:.3f}")
                    st.caption(
                        "R² describes how well LIME's local surrogate fitted the model's perturbed predictions for this run. "
                        "It is not a clinical-validity score and has no universal pass/fail cutoff."
                    )
                else:
                    st.info("LIME did not provide a local surrogate score for this run.")
                st.warning(
                    "This overlay is a model explanation, not a tumour segmentation. "
                    "A plausible-looking image alone cannot confirm that the explanation is reliable. "
                    "Review its local R², repeatability and masking sensitivity, and validate the method on a dataset before making research claims."
                )

                with st.expander("Optional: masking sensitivity test"):
                    st.write("Masks the five most influential regions and compares the selected class score. This runs only when requested.")
                    if st.button("Run masking test", key="run_masking"):
                        try:
                            with st.spinner("Running masking sensitivity test..."):
                                original_score, masked_score = masking_test(
                                    result["input_image"].astype(np.float32), result["segments"],
                                    result["weights"], result["predicted_index"]
                                )
                            st.session_state["masking_result"] = {
                                "image_hash": image_hash, "original": original_score, "masked": masked_score
                            }
                        except Exception as exc:
                            st.error(f"Masking test failed: {exc}")
                    mask_result = st.session_state.get("masking_result")
                    if mask_result and mask_result.get("image_hash") == image_hash:
                        st.metric("Original class score", f"{mask_result['original'] * 100:.2f}%")
                        st.metric("Score after masking", f"{mask_result['masked'] * 100:.2f}%")
                        change = mask_result["masked"] - mask_result["original"]
                        st.write(f"**Change after masking:** {change * 100:+.2f} percentage points")

                with st.expander("Optional: explanation stability test"):
                    st.write("Generates a second explanation using a different random seed. This extra test may take additional time.")
                    if st.button("Check explanation stability", key="check_stability"):
                        try:
                            with st.spinner("Computing a second LIME explanation..."):
                                overlap = stability_test(
                                    result["input_image"].astype(np.float32), result["predicted_index"],
                                    result["samples"], result["weights"]
                                )
                            st.session_state["stability_result"] = {"image_hash": image_hash, "overlap": overlap}
                        except Exception as exc:
                            st.error(f"Stability test failed: {exc}")
                    stability = st.session_state.get("stability_result")
                    if stability and stability.get("image_hash") == image_hash:
                        overlap = stability["overlap"]
                        st.metric("Top-region Jaccard overlap", f"{overlap:.2f}")
                        if overlap >= 0.7:
                            st.success("The top-region overlap is relatively high for these two runs.")
                        else:
                            st.warning("The explanation changed noticeably; consider this result unstable.")
        except Exception as exc:
            st.error("Unable to process the uploaded MRI image.")
            with st.expander("Technical details"):
                st.exception(exc)
    else:
        st.markdown(
            """<div class="soft-card" style="text-align:center;"><div style="font-size:5rem;">🧠</div>
            <h3>Your MRI exploration starts here!</h3><p>Upload an image to see the predicted class and generate a LIME explanation.</p>
            <div style="font-size:2.5rem;">🩺 🤎 🔬</div></div>""",
            unsafe_allow_html=True,
        )

# ============================================================
# 9. ANATOMY GARDEN
# ============================================================
with tab_anatomy:
    st.markdown("## 🫀 Anatomy Garden")
    st.write("Choose an anatomy card to reveal a short educational fact.")
    organs = {
        "🧠 Brain": "The brain processes information and supports memory, learning, sensation and movement.",
        "❤️ Heart": "The heart pumps blood through the circulatory system, delivering oxygen and nutrients.",
        "🫁 Lungs": "The lungs exchange oxygen and carbon dioxide during breathing.",
        "🫘 Kidneys": "The kidneys filter blood and help regulate fluid and electrolyte balance.",
        "🧬 DNA": "DNA stores hereditary information used in the development and functioning of living organisms.",
        "🦴 Skeleton": "The skeleton supports the body, protects organs and works with muscles for movement.",
        "🫀 Liver": "The liver processes nutrients, produces bile and metabolises many substances.",
        "🦷 Teeth": "Teeth help break down food mechanically during digestion.",
    }
    if "selected_organ" not in st.session_state:
        st.session_state["selected_organ"] = "🧠 Brain"
    organ_items = list(organs.items())
    for start in range(0, len(organ_items), 4):
        columns = st.columns(4)
        for col, (organ, fact) in zip(columns, organ_items[start:start + 4]):
            with col:
                st.markdown(f'<div class="soft-card" style="text-align:center;min-height:105px;"><h3>{organ}</h3></div>', unsafe_allow_html=True)
                if st.button("Explore ✨", key=f"organ_{organ}", use_container_width=True):
                    st.session_state["selected_organ"] = organ
    selected = st.session_state["selected_organ"]
    st.markdown(
        f'<div class="hero"><div class="hero-art">{selected.split(" ")[0]}</div><h2>{selected}</h2><p>{organs[selected]}</p></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# 10. MODEL PERFORMANCE
# ============================================================
with tab_performance:
    st.markdown("## 📊 Model evaluation")
    st.write("This section displays saved evaluation files when available. It does not invent performance metrics.")
    accuracy_path = os.path.join(RESULTS_DIR, "accuracy.png")
    loss_path = os.path.join(RESULTS_DIR, "loss.png")
    confusion_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            st.markdown("### Recorded metrics")
            if isinstance(metrics, dict) and metrics:
                items = list(metrics.items())
                for start in range(0, len(items), 3):
                    columns = st.columns(3)
                    for col, (name, value) in zip(columns, items[start:start + 3]):
                        if isinstance(value, (int, float)):
                            display = f"{value * 100:.2f}%" if 0 <= value <= 1 else f"{value:.4g}"
                        else:
                            display = str(value)
                        with col:
                            st.metric(str(name).replace("_", " ").title(), display)
            else:
                st.json(metrics)
        except Exception as exc:
            st.warning("Could not read metrics.json.")
            st.exception(exc)
    else:
        st.info("metrics.json was not found in the brain_tumor_results folder.")
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.markdown("### 📈 Accuracy curve")
        if os.path.exists(accuracy_path):
            st.image(accuracy_path, use_container_width=True)
        else:
            st.info("accuracy.png was not found.")
    with graph_col2:
        st.markdown("### 📉 Loss curve")
        if os.path.exists(loss_path):
            st.image(loss_path, use_container_width=True)
        else:
            st.info("loss.png was not found.")
    st.markdown("### 🧮 Confusion matrix")
    if os.path.exists(confusion_path):
        st.image(confusion_path, use_container_width=True)
    else:
        st.info("confusion_matrix.png was not found.")
    st.divider()
    st.markdown("### VGG16 versus VGG16 + LIME")
    st.markdown(
        """<div class="soft-card"><h3>VGG16</h3><p>Produces the classification output for an MRI image.</p>
        <hr><h3>VGG16 + LIME</h3><p>Uses the same classifier and additionally explains which image regions influenced an individual prediction.</p></div>""",
        unsafe_allow_html=True,
    )
    st.caption("LIME does not automatically retrain VGG16 or improve classification accuracy. Compare classification metrics on the same held-out dataset and report explanation quality separately.")

# ============================================================
# 11. ABOUT
# ============================================================
with tab_about:
    st.markdown("## 📘 About Brain Bloom")
    st.markdown(
        """<div class="hero"><div class="hero-art">🧠 🤎 🩺</div><h2>Making AI easier to understand</h2>
        <p>Brain Bloom demonstrates brain MRI classification with VGG16 and local explanations with LIME.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("### 🎯 Project objectives")
    st.markdown("""
    - Explore brain MRI image classification.
    - Display the predicted class and model output scores.
    - Visualize influential image regions with LIME.
    - Demonstrate optional masking and explanation stability tests.
    - Encourage careful interpretation of AI outputs.
    """)
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown('<div class="soft-card"><h3>🧠 Deep learning</h3><p>TensorFlow and Keras</p><p>VGG16 image classification</p><p>NumPy image processing</p></div>', unsafe_allow_html=True)
    with tech_col2:
        st.markdown('<div class="soft-card"><h3>🔎 Explainable AI</h3><p>LIME image explanations</p><p>Superpixel segmentation</p><p>Streamlit interface</p></div>', unsafe_allow_html=True)
    st.markdown("### 🩺 Responsible AI")
    st.write("This is an educational demonstration, not a clinically validated diagnostic system. Its predictions and explanations may be wrong or unstable.")
    st.markdown('<div class="disclaimer"><b>For research and learning only.</b> Consult qualified healthcare professionals for medical interpretation or diagnosis.</div>', unsafe_allow_html=True)

# ============================================================
# 12. FOOTER
# ============================================================
st.divider()
st.markdown(
    """<div style="text-align:center;padding:18px 8px 30px;color:#705D53;">
    <div style="font-size:2.2rem;">🧠 🤎 🩺</div><h3 style="margin-bottom:6px;">Brain Bloom</h3>
    <p>Explore the image. Understand the prediction. Keep people at the centre of healthcare.</p>
    <p style="font-size:.85rem;">Educational AI demonstration · VGG16 + LIME</p></div>""",
    unsafe_allow_html=True,
)
