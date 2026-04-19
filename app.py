"""Premium Streamlit interface for SMS spam detection."""

import json
import pickle
from pathlib import Path

import streamlit as st


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --glass: rgba(255, 255, 255, 0.13);
            --border: rgba(255, 255, 255, 0.24);
            --text: #f8fbff;
            --muted: rgba(248, 251, 255, 0.72);
            --green: #24d18f;
            --red: #ff4d67;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 18% 18%, rgba(36, 209, 143, 0.32), transparent 28%),
                radial-gradient(circle at 82% 20%, rgba(255, 77, 103, 0.28), transparent 28%),
                linear-gradient(125deg, #09111f, #121a33, #102b3d, #241536);
            background-size: 180% 180%;
            animation: gradientShift 16s ease infinite;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .block-container {
            max-width: 920px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }

        header, footer, #MainMenu {
            visibility: hidden;
        }

        .hero {
            text-align: center;
            margin-bottom: 1.35rem;
            animation: riseIn 0.7s ease both;
        }

        .brand-mark {
            width: 72px;
            height: 72px;
            display: grid;
            place-items: center;
            margin: 0 auto 1rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid var(--border);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
            backdrop-filter: blur(18px);
            font-size: 2.2rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.35rem, 6vw, 4.4rem);
            font-weight: 800;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 660px;
            margin: 0.8rem auto 0;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .glass-card {
            padding: 1.5rem;
            border: 1px solid var(--border);
            background: var(--glass);
            border-radius: 8px;
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            animation: riseIn 0.8s ease both;
        }

        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.42);
            box-shadow: 0 32px 100px rgba(0, 0, 0, 0.42);
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stTextArea textarea {
            min-height: 190px;
            color: #f8fbff !important;
            border: 1px solid rgba(255, 255, 255, 0.26) !important;
            border-radius: 8px !important;
            background: rgba(3, 8, 18, 0.42) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }

        .stTextArea textarea:focus {
            border-color: rgba(36, 209, 143, 0.9) !important;
            box-shadow: 0 0 0 3px rgba(36, 209, 143, 0.18) !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.3rem;
            border: 0;
            border-radius: 8px;
            color: #07111f;
            background: linear-gradient(135deg, #24d18f, #79f2c8);
            font-weight: 800;
            letter-spacing: 0;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
            box-shadow: 0 18px 44px rgba(36, 209, 143, 0.28);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            color: #07111f;
            filter: brightness(1.06);
            box-shadow: 0 24px 58px rgba(36, 209, 143, 0.42);
        }

        .example-title {
            color: var(--muted);
            font-weight: 700;
            margin: 1.15rem 0 0.55rem;
        }

        div[data-testid="column"] .stButton > button {
            min-height: 2.6rem;
            color: #f8fbff;
            background: rgba(255, 255, 255, 0.11);
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: none;
            font-weight: 650;
        }

        div[data-testid="column"] .stButton > button:hover {
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.38);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        }

        .result-card {
            margin-top: 1.35rem;
            padding: 1.25rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.22);
            animation: popIn 0.45s ease both;
            box-shadow: 0 22px 70px rgba(0, 0, 0, 0.34);
        }

        .result-spam {
            background: linear-gradient(135deg, rgba(255, 77, 103, 0.92), rgba(126, 28, 48, 0.86));
        }

        .result-ham {
            background: linear-gradient(135deg, rgba(36, 209, 143, 0.9), rgba(12, 111, 92, 0.86));
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.96) translateY(10px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        .result-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .result-title {
            font-size: 1.7rem;
            font-weight: 850;
            margin: 0;
        }

        .result-emoji {
            font-size: 2.4rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .metric {
            padding: 0.9rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .metric span {
            display: block;
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .metric strong {
            display: block;
            margin-top: 0.25rem;
            font-size: 1.18rem;
        }

        .warning {
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255, 184, 77, 0.15);
            border: 1px solid rgba(255, 184, 77, 0.36);
            color: #fff3d6;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .metric-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, None

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)

    metadata = {"best_model": model.__class__.__name__}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, vectorizer, metadata


def predict_message(message: str, model, vectorizer) -> tuple[int, float]:
    features = vectorizer.transform([message])
    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])
    else:
        confidence = 1.0

    return prediction, confidence


def set_example(message: str) -> None:
    st.session_state["message"] = message


inject_css()

if "message" not in st.session_state:
    st.session_state["message"] = ""

st.markdown(
    """
    <section class="hero">
        <div class="brand-mark">🛡️</div>
        <h1>SpamShield AI</h1>
        <p>Instantly classify SMS messages with a polished ML pipeline trained on TF-IDF n-grams and automatically selected models.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

message = st.text_area(
    "Message",
    key="message",
    placeholder="Paste an SMS here, for example: Congratulations! You have won a free prize. Claim now...",
    label_visibility="collapsed",
)

st.markdown('<div class="example-title">Try an example</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.button(
        "🚫 Prize alert",
        on_click=set_example,
        args=("Congratulations! You have won a $1000 gift card. Reply WIN now to claim your prize.",),
    )
with col2:
    st.button(
        "✅ Friendly check-in",
        on_click=set_example,
        args=("Hey, are we still meeting for coffee at 6 today?",),
    )
with col3:
    st.button(
        "🚫 Urgent offer",
        on_click=set_example,
        args=("URGENT! Your account has been selected for a limited cash reward. Click the link now.",),
    )

model, vectorizer, metadata = load_artifacts()
predict_clicked = st.button("Analyze Message")

if model is None or vectorizer is None:
    st.markdown(
        """
        <div class="warning">
            Model artifacts were not found. Place <strong>spam.csv</strong> in this folder and run
            <strong>python model.py --data spam.csv</strong>, then refresh this app.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif predict_clicked:
    if not message.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        with st.spinner("Scanning message patterns..."):
            prediction, confidence = predict_message(message, model, vectorizer)

        is_spam = prediction == 1
        card_class = "result-spam" if is_spam else "result-ham"
        title = "Spam Detected" if is_spam else "Not Spam"
        emoji = "🚫" if is_spam else "✅"
        model_name = metadata.get("best_model", model.__class__.__name__)

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-top">
                    <p class="result-title">{title}</p>
                    <div class="result-emoji">{emoji}</div>
                </div>
                <div class="metric-row">
                    <div class="metric">
                        <span>Confidence</span>
                        <strong>{confidence * 100:.2f}%</strong>
                    </div>
                    <div class="metric">
                        <span>Model Used</span>
                        <strong>{model_name}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
