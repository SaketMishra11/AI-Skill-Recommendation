
import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Skill Recommendation System",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #101820;
        color: #F5F7F8;
    }

    h1, h2, h3 {
        color: #40D99A !important;
    }

    [data-testid="stMetric"] {
        background-color: #1B2933;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #30434D;
    }

    .stButton > button {
        background-color: #176B48;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #21865B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 AI Skill Recommendation System")

st.write(
    "Discover suitable career paths based on your technical skills."
)

st.divider()


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    # This file contains CSV data despite its .xls extension.
    df = pd.read_csv("new_dataset.xls")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required_columns = {"job_role", "skills"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'job_role' and 'skills' columns."
        )

    df["job_role"] = (
        df["job_role"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["skills"] = (
        df["skills"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove empty records
    df = df[
        (df["job_role"] != "") &
        (df["skills"] != "")
    ].copy()

    if df.empty:
        raise ValueError(
            "Dataset contains no valid job roles or skills."
        )

    return df


# ==========================================
# 3. TRAIN MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def train_model():

    df = load_data()

    X_text = df["skills"]
    y = df["job_role"]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two job roles."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Every job role must have at least two records "
            "for stratified train-test splitting."
        )

    # Split dataset into training and testing data
    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            X_text,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Convert textual skills into numerical TF-IDF features
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return df, vectorizer, model, accuracy


# ==========================================
# 4. CLEAN TEXT
# ==========================================

def clean_text(text):

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    text = text.replace(
        "machinelearning",
        "machine learning"
    )

    # Keep Power BI consistent with the dataset vocabulary
    text = text.replace(
        "power bi",
        "powerbi"
    )

    return text


# ==========================================
# 5. SKILL EXTRACTION
# ==========================================

MULTIWORD_SKILLS = [
    "natural language processing",
    "business intelligence",
    "machine learning",
    "deep learning",
    "computer vision",
    "data visualization",
    "data analysis",
    "data science",
    "web development",
    "power bi",
    "ui/ux",
    "front end",
    "back end"
]


def extract_skills(text):

    text = clean_text(text)

    # Preserve common multiword skills as single tokens
    for skill in sorted(
        MULTIWORD_SKILLS,
        key=len,
        reverse=True
    ):

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        text = re.sub(
            pattern,
            skill.replace(" ", "_"),
            text
        )

    # Split skill lists on spaces and common separators
    tokens = re.split(
        r"[\s,;|/]+",
        text
    )

    return {
        token.strip(".,:()[]{}")
        for token in tokens
        if token.strip(".,:()[]{}")
    }


# ==========================================
# 6. RECOMMEND TOP 3 JOBS
# ==========================================

def recommend_top_jobs(
    user_skills,
    vectorizer,
    model
):

    user_skills = clean_text(user_skills)

    skills_vector = vectorizer.transform(
        [user_skills]
    )

    probabilities = model.predict_proba(
        skills_vector
    )[0]

    # Select the three highest model scores
    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    recommendations = []

    for i in top_indices:

        recommendations.append({
            "job": model.classes_[i],
            "score": float(probabilities[i])
        })

    return recommendations


# ==========================================
# 7. GET UNIQUE REQUIRED SKILLS
# ==========================================

def get_required_skills(job, df):

    rows = df.loc[
        df["job_role"].str.lower() == job.lower(),
        "skills"
    ]

    if rows.empty:
        return []

    required = set()

    # Combine skills from records for this role
    for text in rows:
        required.update(
            extract_skills(text)
        )

    # Convert underscores to readable spaces
    return sorted(
        skill.replace("_", " ")
        for skill in required
    )


# ==========================================
# 8. SKILL GAP ANALYSIS
# ==========================================

def skill_gap(user_skills, job, df):

    required = set(
        get_required_skills(job, df)
    )

    user = {
        skill.replace("_", " ")
        for skill in extract_skills(user_skills)
    }

    matched = sorted(
        required.intersection(user)
    )

    missing = sorted(
        required.difference(user)
    )

    return matched, missing


# ==========================================
# 9. MAIN WEBSITE
# ==========================================

try:

    df, vectorizer, model, accuracy = train_model()

    # --------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------

    st.header("📊 Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        len(df)
    )

    col2.metric(
        "Available Job Roles",
        df["job_role"].nunique()
    )

    col3.metric(
        "Test Accuracy",
        f"{accuracy:.2%}"
    )

    st.caption(
        "Accuracy is measured on held-out test data. "
        "It does not guarantee real-world job suitability."
    )

    st.divider()

    # --------------------------------------
    # USER SKILLS INPUT
    # --------------------------------------

    st.header("🔍 Find Your Best Career")

    user_skills = st.text_area(
        "Enter your technical skills",
        placeholder=(
            "Example: Python, SQL, machine learning, "
            "Pandas, Power BI, Excel"
        ),
        height=120
    )

    # Store submitted skills across Streamlit reruns
    if "submitted_skills" not in st.session_state:
        st.session_state.submitted_skills = ""

    if st.button(
        "Recommend Jobs",
        type="primary"
    ):

        if not user_skills.strip():

            st.warning(
                "Please enter your skills first."
            )

        else:

            st.session_state.submitted_skills = user_skills

    # --------------------------------------
    # RECOMMENDATIONS AND SKILL GAP
    # --------------------------------------

    submitted_skills = st.session_state.submitted_skills

    if submitted_skills:

        recommendations = recommend_top_jobs(
            submitted_skills,
            vectorizer,
            model
        )

        st.divider()

        st.header("🎯 Top 3 Recommended Jobs")

        rec_columns = st.columns(
            len(recommendations)
        )

        for rank, (column, item) in enumerate(
            zip(rec_columns, recommendations),
            start=1
        ):

            with column:

                with st.container(border=True):

                    st.markdown(
                        f"### {rank}. {item['job']}"
                    )

                    st.progress(
                        item["score"]
                    )

                    st.write(
                        f"Model score: {item['score']:.2%}"
                    )

        st.caption(
            "These scores are classifier outputs, not calibrated "
            "probabilities of getting hired."
        )

        st.divider()

        # ----------------------------------
        # SKILL GAP ANALYSIS
        # ----------------------------------

        st.header("📚 Skill Gap Analysis")

        job_options = sorted(
            df["job_role"].unique().tolist()
        )

        default_job = recommendations[0]["job"]

        default_index = job_options.index(
            default_job
        )

        selected_job = st.selectbox(
            "Select a job role",
            job_options,
            index=default_index,
            key="skill_gap_job"
        )

        required = get_required_skills(
            selected_job,
            df
        )

        matched, missing = skill_gap(
            submitted_skills,
            selected_job,
            df
        )

        # Skill lists displayed in separate columns
        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:

            st.subheader("✅ Matched Skills")

            if matched:

                for skill in matched:
                    st.write(f"✅ {skill}")

            else:

                st.info(
                    "No matching skills detected."
                )

        with skill_col2:

            st.subheader("📌 Skills to Learn")

            if missing:

                for skill in missing:
                    st.write(f"📖 {skill}")

            else:

                st.success(
                    "All listed skills matched!"
                )

        st.divider()

        # Skill statistics
        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Required Skills",
            len(required)
        )

        metric2.metric(
            "Matched Skills",
            len(matched)
        )

        metric3.metric(
            "Missing Skills",
            len(missing)
        )

        if required:

            coverage = (
                len(matched) / len(required)
            )

            st.subheader("📈 Skill Coverage")

            st.progress(coverage)

            st.write(
                f"Matched skill coverage: {coverage:.1%}"
            )

        st.info(
            "Skill gaps are calculated using unique skill terms "
            "in your dataset. Synonyms, experience and proficiency "
            "levels are not evaluated."
        )

    st.divider()

    # --------------------------------------
    # JOB ROLE DISTRIBUTION
    # --------------------------------------

    st.header("📊 Job Role Distribution")

    job_counts = df["job_role"].value_counts()

    st.bar_chart(job_counts)

    # --------------------------------------
    # DATASET PREVIEW
    # --------------------------------------

    with st.expander("📁 View Dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )


# ==========================================
# 10. ERROR HANDLING
# ==========================================

except FileNotFoundError:

    st.error(
        "Dataset not found. Keep new_dataset.xls "
        "in the same folder as app.py."
    )

except ImportError as e:

    st.error(
        f"Required library missing: {e}. "
        "Install dependencies in the VS Code terminal."
    )

except Exception as e:

    st.error(
        f"Unable to start the app: {e}"
    )
