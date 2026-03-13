import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Student GPA Predictor", page_icon="🎓", layout="wide")

# ==========================
# Load Model (cached)
# ==========================
@st.cache_resource
def load_artifacts():
    model_obj = pickle.load(open("knn_model.pkl", "rb"))
    scaler_obj = pickle.load(open("scaler.pkl", "rb"))
    return model_obj, scaler_obj

model, scaler = load_artifacts()

# ==========================
# Custom UI Styling
# ==========================
st.markdown(
    """
    <style>
    :root {
        --bg1: #0b1020;
        --bg2: #111827;
        --card: #111a2e;
        --stroke: #22304d;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --primary: #60a5fa;
        --success: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1d4ed8 0%, var(--bg2) 35%, var(--bg1) 100%);
        color: var(--text);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #93c5fd, #60a5fa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .card {
        background: linear-gradient(180deg, #0f172a 0%, var(--card) 100%);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }

    .card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        color: #bfdbfe;
    }

    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        border: 1px solid #334155;
        margin-right: 8px;
        margin-bottom: 8px;
        color: #e2e8f0;
        background: #1e293b;
    }

    .pill-yes { background: rgba(34, 197, 94, 0.18); border-color: rgba(34, 197, 94, 0.5); }
    .pill-no  { background: rgba(239, 68, 68, 0.16); border-color: rgba(239, 68, 68, 0.45); }

    .status-ok { color: #86efac; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Header
# ==========================
st.markdown('<div class="main-title">🎓 Student GPA Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict student GPA using <b>KNN Regression</b> with <b>StandardScaler</b>.</div>',
    unsafe_allow_html=True
)

# ==========================
# Sidebar Inputs
# ==========================
st.sidebar.header("Student Information")
st.sidebar.caption("Binary fields use **Yes = 1** and **No = 0**")

with st.sidebar.form("student_form"):
    Absences = st.slider("Absences", 0, 30, 5)
    GradeClass = st.selectbox("Grade Class", [0, 1, 2, 3])
    ParentalSupport = st.selectbox("Parental Support", [0, 1, 2, 3, 4])
    StudyTimeWeekly = st.slider("Study Time Weekly (hours)", 0, 20, 5)

    Tutoring = 1 if st.selectbox("Tutoring", ["No", "Yes"]) == "Yes" else 0
    Extracurricular = 1 if st.selectbox("Extracurricular", ["No", "Yes"]) == "Yes" else 0
    Music = 1 if st.selectbox("Music", ["No", "Yes"]) == "Yes" else 0
    Sports = 1 if st.selectbox("Sports", ["No", "Yes"]) == "Yes" else 0

    submitted = st.form_submit_button("Predict GPA 🎯", use_container_width=True)

# ==========================
# Main Layout
# ==========================
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card"><h4>📌 Input Summary</h4>', unsafe_allow_html=True)

    tutoring_text = "Yes" if Tutoring == 1 else "No"
    extra_text = "Yes" if Extracurricular == 1 else "No"
    music_text = "Yes" if Music == 1 else "No"
    sports_text = "Yes" if Sports == 1 else "No"

    tutoring_cls = "pill-yes" if Tutoring == 1 else "pill-no"
    extra_cls = "pill-yes" if Extracurricular == 1 else "pill-no"
    music_cls = "pill-yes" if Music == 1 else "pill-no"
    sports_cls = "pill-yes" if Sports == 1 else "pill-no"

    st.markdown(
        f"""
        <span class="pill">Absences: {Absences}</span>
        <span class="pill">GradeClass: {GradeClass}</span>
        <span class="pill">ParentalSupport: {ParentalSupport}</span>
        <span class="pill">StudyTimeWeekly: {StudyTimeWeekly}h</span>
        <span class="pill {tutoring_cls}">Tutoring: {tutoring_text}</span>
        <span class="pill {extra_cls}">Extracurricular: {extra_text}</span>
        <span class="pill {music_cls}">Music: {music_text}</span>
        <span class="pill {sports_cls}">Sports: {sports_text}</span>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h4>🧠 Model Status</h4>', unsafe_allow_html=True)
    st.write("Algorithm: **KNN Regressor**")
    st.write("Scaling: **StandardScaler**")
    st.markdown('<span class="status-ok">● Ready for prediction</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# Prediction
# ==========================
if submitted:
    # Exact feature order used during training
    features = np.array([[
        Absences, GradeClass, ParentalSupport, StudyTimeWeekly,
        Tutoring, Extracurricular, Music, Sports
    ]], dtype=float)

    expected = getattr(scaler, "n_features_in_", features.shape[1])
    if features.shape[1] != expected:
        st.error(f"Feature mismatch: model expects {expected}, app sends {features.shape[1]}.")
        st.stop()

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    pred_value = float(prediction[0])

    # Clamp to GPA range for display
    pred_display = max(0.0, min(4.0, pred_value))
    progress_value = int((pred_display / 4.0) * 100)

    st.markdown("### ✅ Prediction Result")
    st.success(f"Predicted GPA: **{pred_display:.2f} / 4.00**")
    st.progress(progress_value, text=f"GPA Strength: {progress_value}%")

    if pred_display >= 3.5:
        st.balloons()
        st.info("Excellent performance 🚀")
        st.write("Keep consistency in study schedule and co-curricular balance.")
    elif pred_display >= 2.5:
        st.info("Good performance 👍")
        st.write("Small improvements in attendance and weekly study time can raise GPA.")
    else:
        st.warning("Needs improvement 📚")
        st.write("Focus on reducing absences, increasing study hours, and using tutoring support.")

st.markdown("---")
st.caption("Built with Python • Scikit-Learn • Streamlit")