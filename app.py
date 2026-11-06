import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_spam_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "spam_history.db"

CONFUSION_MATRIX_PATH = BASE_DIR / "models" / "confusion_matrix.png"

ASSETS_DIR = BASE_DIR / "assets"

MOHIT_PHOTO = ASSETS_DIR / "mohit_kumar.jpg"
PUSHPA_PHOTO = ASSETS_DIR / "pushpa_kumari.jpg"
DEVELOPERS_DIR = BASE_DIR / "assets" / "developers"

MOHIT_PHOTO = DEVELOPERS_DIR / "mohit_kumar.jpg"
PUSHPA_PHOTO = DEVELOPERS_DIR / "pushpa_kumari.jpg"

# =====================================================
# 2. DEVELOPER INFORMATION
# =====================================================

DEVELOPERS = [
    {
        "name": "Mohit Kumar",
        "role": "AI & ML Developer",
        "email": "mk7469112@gmail.com",
        "photo": MOHIT_PHOTO,
        "about": (
            "B.Tech Artificial Intelligence and Machine Learning "
            "student interested in Python programming, IoT, "
            "Arduino, and Embedded Systems. Passionate about "
            "building practical technology solutions."
        ),
        "skills": [
            "Python",
            "C",
            "C++",
            "Arduino",
            "IoT",
            "Embedded Systems",
            "SQL",
            "NumPy",
            "Pandas"
        ],
        "linkedin": "https://www.linkedin.com/in/mohit-kumar-07to18",
        "github": "https://github.com/Mohitkumar-lan",
        "instagram": "https://www.instagram.com/mr_mohit_guptaaa?xtok=dTkyb2gyOG52Z3Rv"
    },
    {
        "name": "Pushpa Kumari",
        "role": "AI/ML & Python Developer",
        "email": "pushpastm8252@gmail.com",
        "photo": PUSHPA_PHOTO,
        "about": (
            "B.Tech Computer Science student specializing in "
            "Artificial Intelligence and Machine Learning. "
            "Interested in Python development, Machine Learning, "
            "Generative AI, RAG, and building real-world projects."
        ),
        "skills": [
            "Python",
            "Java",
            "C",
            "C++",
            "Machine Learning",
            "Generative AI",
            "RAG",
            "Streamlit",
            "SQL",
            "GitHub"
        ],
        "linkedin": "https://www.linkedin.com/in/pushpa-kumari-302184380",
        "github": "https://github.com/pushpastm8252-debug",
        "instagram": "https://www.instagram.com/pushpa74800?srtk=NGQyMjQ4bDQxa28z"
    }
]


# =====================================================
# 3. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 4. PROFESSIONAL DARK UI
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #08142F !important;
    color: #F8FAFC !important;
}

[data-testid="stHeader"] {
    background: #08142F !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1550px;
}

.stApp p,
.stApp label,
.stApp span,
.stApp li {
    color: #E2E8F0;
}

h1, h2, h3, h4 {
    color: #F8FAFC !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1938, #071229) !important;
    border-right: 1px solid #263B68;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 9px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 10px;
    padding: 10px;
    color: #DCE7FF !important;
}

/* Hero */

.hero {
    background: linear-gradient(
        115deg,
        #0877F9 0%,
        #123CBA 55%,
        #3216B9 100%
    );
    border: 1px solid #315BDE;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 20px;
    box-shadow: 0 12px 35px rgba(0, 80, 255, 0.18);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #E3EEFF !important;
    font-size: 17px;
    margin-top: 10px;
}

/* Metric cards */

.metric-card {
    border-radius: 17px;
    padding: 20px;
    min-height: 112px;
    border: 1px solid #31518A;
    background: linear-gradient(135deg, #153D85, #0B2350);
}

.metric-green {
    background: linear-gradient(135deg, #087F70, #064A49);
    border-color: #0DAF91;
}

.metric-red {
    background: linear-gradient(135deg, #9C294D, #501A3A);
    border-color: #E34C71;
}

.metric-purple {
    background: linear-gradient(135deg, #6030BD, #32166F);
    border-color: #8B5CF6;
}

.metric-label {
    color: #D9E7FF !important;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 29px;
    font-weight: 800;
    margin-top: 7px;
}

/* Panels */

.panel {
    background: linear-gradient(145deg, #101F40, #0B1732);
    border: 1px solid #304675;
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 18px;
}

.panel-title {
    color: #F8FAFC !important;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 15px;
}

/* Inputs */

.stTextArea textarea {
    background: #111F40 !important;
    color: #FFFFFF !important;
    border: 1px solid #50699F !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}

.stTextArea textarea::placeholder {
    color: #9AAAD0 !important;
}

/* Buttons */

.stButton button,
.stDownloadButton button,
.stLinkButton a {
    background: linear-gradient(90deg, #168CF9, #1455EF) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 42px;
}

/* Result */

.result-spam {
    background: linear-gradient(110deg, #9E294D, #6D1B43);
    border: 1px solid #F43F65;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.result-ham {
    background: linear-gradient(110deg, #087F70, #075448);
    border: 1px solid #10B981;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.result-spam h2,
.result-ham h2 {
    color: white !important;
    margin: 0;
}

.result-spam p,
.result-ham p {
    color: #F8FAFC !important;
    margin-bottom: 0;
}

/* Keyword tags */

.keyword-tag {
    display: inline-block;
    background: #8C2852;
    border: 1px solid #D84C7A;
    color: white !important;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

/* Developer cards */

.developer-card {
    background: linear-gradient(145deg, #142B55, #0C1B39);
    border: 1px solid #34558D;
    border-radius: 18px;
    padding: 22px;
    min-height: 100%;
    margin-bottom: 18px;
}

.developer-role {
    color: #65C8FF !important;
    font-weight: 700;
    font-size: 14px;
}

.developer-about {
    color: #D6E3FA !important;
    line-height: 1.7;
    min-height: 105px;
}

.developer-email {
    color: #79BFFF !important;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.skill-tag {
    display: inline-block;
    background: #173D75;
    border: 1px solid #3D72B9;
    color: #EAF4FF !important;
    padding: 6px 10px;
    margin: 4px 3px;
    border-radius: 20px;
    font-size: 12px;
}

.developer-image {
    border: 3px solid #4D91EF;
    border-radius: 16px;
}

/* Footer */

.footer {
    text-align: center;
    color: #91A4CA !important;
    border-top: 1px solid #263B68;
    margin-top: 35px;
    padding: 22px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 5. DATABASE
# =====================================================

def get_connection():

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL

            )
        """)


def save_prediction(message, prediction, confidence):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO prediction_history
            (message, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            message,
            prediction,
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_history():

    with get_connection() as conn:

        return pd.read_sql_query("""
            SELECT
                id,
                message,
                prediction,
                confidence,
                created_at
            FROM prediction_history
            ORDER BY id DESC
        """, conn)


# =====================================================
# 6. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model not found. Run python train_model.py first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():

    if not METRICS_PATH.exists():
        return {}

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =====================================================
# 7. KEYWORD ANALYSIS
# =====================================================

SUSPICIOUS_KEYWORDS = [
    "free", "winner", "won", "prize", "claim",
    "urgent", "congratulations", "cash", "reward",
    "click", "offer", "limited time", "verify",
    "password", "bank account", "lottery", "gift",
    "credit card", "act now"
]


def analyze_keywords(message):

    found = []

    for keyword in SUSPICIOUS_KEYWORDS:

        pattern = (
            r"(?<!\w)"
            + re.escape(keyword)
            + r"(?!\w)"
        )

        if re.search(pattern, message.lower()):
            found.append(keyword)

    return found


# =====================================================
# 8. DEVELOPER PROFILE COMPONENT
# =====================================================

def show_developer(developer):

    st.markdown('<div class="developer-card">', unsafe_allow_html=True)

    photo_path = developer["photo"]

    if photo_path.exists():

        st.image(
            str(photo_path),
            width=210
        )

    else:

        st.markdown(
            f"""
            <div style="
                width:150px;
                height:150px;
                border-radius:50%;
                background:#173D75;
                border:3px solid #4D91EF;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:65px;
                margin-bottom:15px;
            ">
                👤
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            f"Photo not found: assets/{photo_path.name}"
        )

    st.markdown(
        f"<h2>{developer['name']}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="developer-role">{developer["role"]}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="developer-about">{developer["about"]}</p>',
        unsafe_allow_html=True
    )

    st.markdown("**Technical Skills**")

    skill_html = "".join(
        f'<span class="skill-tag">{skill}</span>'
        for skill in developer["skills"]
    )

    st.markdown(
        skill_html,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("**Contact**")

    st.markdown(
        f'<p class="developer-email">✉️ {developer["email"]}</p>',
        unsafe_allow_html=True
    )

    st.link_button(
        "📧 Send Email",
        f"mailto:{developer['email']}",
        use_container_width=True
    )

    social_links = [
        ("LinkedIn", developer["linkedin"]),
        ("GitHub", developer["github"]),
        ("Instagram", developer["instagram"])
    ]

    available_links = [
        (label, url)
        for label, url in social_links
        if url.strip()
    ]

    if available_links:

        st.markdown("**Social Profiles**")

        for label, url in available_links:

            st.link_button(
                f"Visit {label}",
                url,
                use_container_width=True
            )

    else:

        st.caption(
            "Social profile links can be added in the developer settings."
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# 9. LOAD APPLICATION DATA
# =====================================================

initialize_database()

try:

    model = load_model()
    metrics = load_metrics()

except Exception as error:

    st.error(f"Application loading error: {error}")
    st.stop()


# =====================================================
# 10. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0 25px;">
        <div style="font-size:65px;">📧</div>
        <h2 style="color:white !important; margin:0;">
            Email Spam
        </h2>
        <h2 style="color:white !important; margin:0;">
            Detection System
        </h2>
        <p style="color:#AFC5F7 !important;">
            Smart Filter. Safer Inbox.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Analytics",
            "History",
            "About",
            "Developers"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style="
        background:linear-gradient(145deg,#102D68,#091B43);
        border:1px solid #315B9D;
        border-radius:15px;
        padding:20px;
        text-align:center;
        margin-top:25px;
    ">
        <div style="font-size:42px;">🚀</div>
        <h3 style="color:white !important;">
            Powered by Machine Learning
        </h3>
        <p style="color:#C6D9FF !important;">
            Intelligent message classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Model: " + str(metrics.get("best_model", "Unknown"))
    )


# =====================================================
# 11. COMMON DATA
# =====================================================

history = get_history()

total = len(history)

spam_count = (
    int((history["prediction"] == "SPAM").sum())
    if total else 0
)

ham_count = (
    int((history["prediction"] == "NOT SPAM").sum())
    if total else 0
)

spam_percent = spam_count / total * 100 if total else 0
ham_percent = ham_count / total * 100 if total else 0

best_name = metrics.get("best_model", "SVM")

best_accuracy = (
    metrics.get("models", {})
    .get(best_name, {})
    .get("accuracy", 0)
)


# =====================================================
# 12. HEADER
# =====================================================

top1, top2 = st.columns([5, 1])

with top2:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            background:#132C55;
            border:1px solid #35568C;
            padding:10px;
            border-radius:25px;
            color:white;
        ">
            🟢 Model: {best_name}
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 13. HOME PAGE
# =====================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div style="font-size:43px;">🛡️ 📩</div>
        <h1>
            Email <span style="color:#54D5FF;">Spam</span>
            Detection System
        </h1>
        <p>
            Detect spam emails using the power of Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📩 Total Emails</div>
            <div class="metric-value">{total:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card metric-green">
            <div class="metric-label">✅ Not Spam</div>
            <div class="metric-value">{ham_count:,}</div>
            <div class="metric-label">{ham_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card metric-red">
            <div class="metric-label">⚠️ Spam</div>
            <div class="metric-value">{spam_count:,}</div>
            <div class="metric-label">{spam_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card metric-purple">
            <div class="metric-label">🎯 Model Accuracy</div>
            <div class="metric-value">{best_accuracy * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.05, 0.95], gap="medium")

    with left:

        st.markdown(
            '<div class="panel-title">📨 Enter Email Content</div>',
            unsafe_allow_html=True
        )

        message = st.text_area(
            "Email content",
            placeholder=(
                "Paste your email content here...\n"
                "(e.g. Congratulations! You have won a free gift...)"
            ),
            height=170,
            label_visibility="collapsed",
            key="email_input"
        )

        detect = st.button(
            "🔍 Detect Spam",
            type="primary",
            use_container_width=True
        )

        if detect:

            if not message.strip():

                st.warning("Please enter an email message.")

            else:

                try:

                    prediction_value = int(
                        model.predict([message])[0]
                    )

                    probabilities = model.predict_proba(
                        [message]
                    )[0]

                    confidence = float(max(probabilities))

                    prediction = (
                        "SPAM"
                        if prediction_value == 1
                        else "NOT SPAM"
                    )

                    save_prediction(
                        message,
                        prediction,
                        confidence
                    )

                    st.session_state["last_prediction"] = {
                        "message": message,
                        "prediction": prediction,
                        "confidence": confidence
                    }

                    st.rerun()

                except Exception as error:

                    st.error(f"Prediction error: {error}")

    with right:

        st.markdown(
            '<div class="panel-title">📊 Detection Result</div>',
            unsafe_allow_html=True
        )

        if "last_prediction" in st.session_state:

            result = st.session_state["last_prediction"]

            if result["prediction"] == "SPAM":

                st.markdown("""
                <div class="result-spam">
                    <h2>⚠️ SPAM DETECTED</h2>
                    <p>This email looks suspicious.</p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-ham">
                    <h2>✅ NOT SPAM</h2>
                    <p>This message is classified as non-spam.</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Prediction Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.progress(result["confidence"])

            keywords = analyze_keywords(result["message"])

            st.markdown("**Suspicious Keywords**")

            if keywords:

                tags = "".join(
                    f'<span class="keyword-tag">{word}</span>'
                    for word in keywords
                )

                st.markdown(tags, unsafe_allow_html=True)

            else:

                st.success("No predefined keywords detected.")

            st.caption(
                "Model confidence is an estimated probability, "
                "not a guarantee that a message is safe."
            )

        else:

            st.info(
                "Your detection result will appear here after "
                "you analyze a message."
            )

            st.markdown(
                '<div style="text-align:center;padding:25px;font-size:65px;">🛡️</div>',
                unsafe_allow_html=True
            )

    st.write("")

    bottom1, bottom2 = st.columns([1, 1.5], gap="medium")

    with bottom1:

        st.markdown(
            '<div class="panel-title">📊 Spam vs Not Spam Analysis</div>',
            unsafe_allow_html=True
        )

        if total:

            fig, ax = plt.subplots(figsize=(5, 3.4))

            fig.patch.set_facecolor("#101F40")
            ax.set_facecolor("#101F40")

            ax.pie(
                [ham_count, spam_count],
                labels=["Not Spam", "Spam"],
                colors=["#3185F7", "#F43F68"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={
                    "width": 0.42,
                    "edgecolor": "#101F40"
                },
                textprops={
                    "color": "white",
                    "fontsize": 10
                }
            )

            ax.set_aspect("equal")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:

            st.info(
                "Analyze messages to generate your spam distribution chart."
            )

    with bottom2:

        st.markdown(
            '<div class="panel-title">🕒 Recent Predictions</div>',
            unsafe_allow_html=True
        )

        if not history.empty:

            recent = history.head(5).copy()

            recent["message"] = recent["message"].str.slice(0, 42)

            recent["confidence"] = (
                recent["confidence"] * 100
            ).round(1).astype(str) + "%"

            recent = recent.rename(columns={
                "message": "Email Preview",
                "prediction": "Result",
                "confidence": "Confidence",
                "created_at": "Time"
            })

            st.dataframe(
                recent[
                    ["Email Preview", "Result", "Confidence", "Time"]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No predictions yet.")

    st.markdown("""
    <div class="footer">
        Email Spam Detection System |
        Built with Python, NLP and Machine Learning
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 14. ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    st.write(
        "Detailed analysis of your spam detection predictions."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Messages", total)
    c2.metric("Spam Percentage", f"{spam_percent:.2f}%")
    c3.metric("Not Spam Percentage", f"{ham_percent:.2f}%")

    st.divider()

    st.subheader("Machine Learning Model Comparison")

    model_results = metrics.get("models", {})

    if model_results:

        performance_df = pd.DataFrame(model_results).T

        performance_df.index.name = "Model"

        columns = [
            c for c in ["accuracy", "precision", "recall", "f1_score"]
            if c in performance_df.columns
        ]

        st.dataframe(
            performance_df[columns].style.format("{:.2%}"),
            use_container_width=True
        )

        st.subheader("Model Metrics Comparison")

        st.bar_chart(performance_df[columns] * 100)

        st.success(
            f"Best Model: {metrics.get('best_model', 'Unknown')}"
        )

    else:

        st.warning("Model metrics not available.")

    st.divider()

    if CONFUSION_MATRIX_PATH.exists():

        st.subheader("Confusion Matrix")

        st.image(
            str(CONFUSION_MATRIX_PATH),
            use_container_width=True
        )


# =====================================================
# 15. HISTORY PAGE
# =====================================================

elif page == "History":

    st.title("🕒 Prediction History")

    st.write(
        "View and download all messages analyzed by the system."
    )

    history = get_history()

    if history.empty:

        st.info("No prediction history available.")

    else:

        st.metric("Total Predictions", len(history))

        display = history.copy()

        display["confidence"] = (
            display["confidence"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="spam_prediction_history.csv",
            mime="text/csv"
        )


# =====================================================
# 16. ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("📧 About Email Spam Detection System")

    st.markdown("""
    <div class="hero">
        <h1>Smart Filter. Safer Inbox.</h1>
        <p>
            An intelligent message classification system powered by
            Machine Learning and Natural Language Processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Project Objective")

    st.write("""
    The objective of this project is to classify text messages
    into Spam and Not Spam categories using Machine Learning.
    """)

    st.subheader("🛠️ Technologies Used")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF Vectorization
        - Natural Language Processing
        """)

    with c2:

        st.markdown("""
        - Naive Bayes
        - Logistic Regression
        - Support Vector Machine
        - Streamlit
        - SQLite Database
        """)

    st.subheader("🤖 ML Models")

    st.write("""
    The system trains and evaluates three classification algorithms:

    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Support Vector Machine (SVM)

    The best model is selected using the spam-class F1 score.
    """)

    st.info(
        "Dataset note: The current model uses the UCI SMS Spam "
        "Collection dataset. It should be validated on a real email "
        "dataset before claiming email-specific performance."
    )

    st.metric(
        "Best Model Accuracy",
        f"{best_accuracy * 100:.2f}%"
    )


# =====================================================
# 17. DEVELOPERS PAGE
# =====================================================

elif page == "Developers":

    st.markdown("""
    <div class="hero">
        <div style="font-size:42px;">👨‍💻 👩‍💻</div>
        <h1>Meet Our Developers</h1>
        <p>
            The people behind the Email Spam Detection System.
            Passionate about technology, AI and innovation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<h2 style='text-align:center;'>Our Development Team</h2>",
        unsafe_allow_html=True
    )

    st.write(
        "This project is developed by two technology enthusiasts "
        "with an interest in Artificial Intelligence and software development."
    )

    dev1, dev2 = st.columns(2, gap="large")

    with dev1:
        show_developer(DEVELOPERS[0])

    with dev2:
        show_developer(DEVELOPERS[1])

    st.divider()

    st.subheader("🚀 About Our Team")

    st.write("""
    We are students interested in learning and applying modern
    technologies to solve real-world problems.

    Through this project, we explored Machine Learning,
    Natural Language Processing, model evaluation, Streamlit
    application development, and database integration.
    """)

    st.markdown("""
    <div class="footer">
        Designed and Developed with ❤️ by the Development Team
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 18. END
# =====================================================