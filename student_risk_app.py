import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# 🌟 APP CONFIGURATION
# =====================================================
st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓", layout="wide")
st.markdown("<div class='navbar'>STUDENT RISK PREDICTION & ANALYTICS PLATFORM</div>", unsafe_allow_html=True)


# =====================================================
# 🎨 CUSTOM CSS FOR ELEGANT UI
# =====================================================
# =====================================================
# 🎨 PREMIUM MODERN UI CSS (APPLE + GLASS + GRADIENT)
# =====================================================
st.markdown("""
<style>
            /* FULL PAGE BACKGROUND FIX */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #d6eaff, #e8f4ff 50%, #f5fbff) !important;
    background-attachment: fixed;
}


body {
    background: linear-gradient(135deg, #dae7ff, #eef5ff);
    font-family: 'Segoe UI', Arial, sans-serif;
            
}

.block-container {
    padding-top: 2rem !important;
}

/* ======= TOP NAV BAR ======= */
/* ======= TOP NAV BAR ======= */
.navbar {
    background: linear-gradient(90deg, #00204a, #005792);
    padding: 18px;
    border-radius: 12px;
    margin-top: 20px; /* NEW SPACE ADDED */
    margin-bottom: 25px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 1px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}


/* Title Text */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#001f3f,#008cff);
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: 1.5px;
    margin-top: 10px;
}

/* Premium Gradient Cards */
.plot-container, [data-testid="stMetric"], .card {
    background: linear-gradient(145deg, #ffffff, #e3e9f2);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 6px 6px 20px rgba(0,0,0,0.08), -6px -6px 20px rgba(255,255,255,0.7);
    transition: all .3s ease-in-out;
}
            

.plot-container:hover, [data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 8px 8px 30px rgba(0,0,0,0.12);
}

/* Premium Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#e6f3ff,#ffffff);
    padding: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#004aad,#00c8ff);
    color: white;
    padding: 10px 28px;
    border-radius: 30px;
    font-size: 1.1rem;
    border: none;
    font-weight: 600;
    box-shadow: 0px 5px 20px rgba(0,136,255,.4);
    transition: all .3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#002a6a,#0094c8);
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0px 8px 30px rgba(0,136,255,.6);
}

/* Animated TABS */
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    font-weight: 600;
    border-radius: 12px;
    background: #e7f1ff;
    color: #003463;
    margin-right: 6px;
    transition: all .25s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #c6e6ff;
    transform: translateY(-2px);
}

/* Form Section */
.stForm {
    background: rgba(255,255,255,0.70);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.13);
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 🧠 TITLE SECTION
# =====================================================
st.markdown('<div class="main-title">🎓 At-Risk Student Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze student performance and predict academic risk</p>", unsafe_allow_html=True)

# =====================================================
# 📂 LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance_Labeled.csv")

df = load_data()
df.columns = df.columns.str.strip()
df["Risk Level"] = df["Risk Level"].str.strip()

# =====================================================
# 🧹 PREPROCESSING
# =====================================================
target_map = {"Not At-Risk": 0, "At-Risk": 1}
df["Risk Level Encoded"] = df["Risk Level"].map(target_map)

cat_cols = [
    "gender", "race/ethnicity", 
    "parental level of education",
    "lunch", "test preparation course"
]

label_encoders = {}
df_encoded = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# =====================================================
# 🔍 SIDEBAR FILTERS
# =====================================================
st.sidebar.title("🔎 Filters")

gender_f = st.sidebar.selectbox("Gender", ["All"] + sorted(df["gender"].unique()))
race_f = st.sidebar.selectbox("Race/Ethnicity", ["All"] + sorted(df["race/ethnicity"].unique()))
lunch_f = st.sidebar.selectbox("Lunch Type", ["All"] + sorted(df["lunch"].unique()))
prep_f = st.sidebar.selectbox("Test Preparation", ["All"] + sorted(df["test preparation course"].unique()))

math_range = st.sidebar.slider("Math Score", int(df["math score"].min()), int(df["math score"].max()),
                               (int(df["math score"].min()), int(df["math score"].max())))

reading_range = st.sidebar.slider("Reading Score", int(df["reading score"].min()), int(df["reading score"].max()),
                                  (int(df["reading score"].min()), int(df["reading score"].max())))

writing_range = st.sidebar.slider("Writing Score", int(df["writing score"].min()), int(df["writing score"].max()),
                                  (int(df["writing score"].min()), int(df["writing score"].max())))

avg_range = st.sidebar.slider("Average Score", int(df["Average Score"].min()), int(df["Average Score"].max()),
                              (int(df["Average Score"].min()), int(df["Average Score"].max())))

# =====================================================
# 📊 FILTER DATA
# =====================================================
filtered_df = df.copy()

if gender_f != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_f]
if race_f != "All":
    filtered_df = filtered_df[filtered_df["race/ethnicity"] == race_f]
if lunch_f != "All":
    filtered_df = filtered_df[filtered_df["lunch"] == lunch_f]
if prep_f != "All":
    filtered_df = filtered_df[filtered_df["test preparation course"] == prep_f]

filtered_df = filtered_df[
    (filtered_df["math score"].between(*math_range)) &
    (filtered_df["reading score"].between(*reading_range)) &
    (filtered_df["writing score"].between(*writing_range)) &
    (filtered_df["Average Score"].between(*avg_range))
]

# =====================================================
# 📌 KPI CARDS
# =====================================================
st.markdown("### 📊 Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("📘 Total Students", len(filtered_df))
with c2:
    risk_pct = (filtered_df["Risk Level"] == "At-Risk").mean() * 100
    st.metric("🚨 At-Risk %", f"{risk_pct:.1f}%")
with c3:
    st.metric("📚 Avg Score", f"{filtered_df['Average Score'].mean():.1f}")

# =====================================================
# 📉 TABS SECTION
# =====================================================
tabs = st.tabs(["📘 Academic Insights", "👥 Demographic Insights", "📊 Correlation"])

# ---------- Academic Charts ----------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(filtered_df, x="Average Score", hue="Risk Level", multiple="stack", ax=ax)
        ax.set_title("Average Score Distribution")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x="Risk Level", y="Average Score", ax=ax)
        ax.set_title("Average Score vs Risk Level")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Demographics ----------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x="gender", hue="Risk Level", ax=ax)
        ax.set_title("Gender vs Risk Level")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x="race/ethnicity", hue="Risk Level", ax=ax)
        ax.set_title("Race/Ethnicity vs Risk Level")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Correlation ----------
with tabs[2]:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

st.success(f"✅ Showing {len(filtered_df)} filtered student records")
import plotly.express as px
import plotly.graph_objects as go



extra_tabs = st.tabs([
    "📊 Pie & Donut Charts",
    "📈 Line & Area Charts",
    "🔗 Scatter & Bubble Charts",
    "🏆 Treemap & Funnel",
    "🚀 Waterfall & Radar Charts"
])

# ================= PIE CHART / DONUT =================
with extra_tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(filtered_df, names="Risk Level", title="Risk Distribution (Pie)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(filtered_df, names="gender", title="Gender Split (Donut)", hole=.5)
        st.plotly_chart(fig, use_container_width=True)

# ================= LINE / AREA =================
with extra_tabs[1]:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(filtered_df.sort_values("math score"), y="math score", title="Line Chart - Math Score Trend")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.area(filtered_df.sort_values("Average Score"), y="Average Score", title="Area Chart - Average Score Rise/Fall")
        st.plotly_chart(fig, use_container_width=True)

# ================= SCATTER / BUBBLE =================
with extra_tabs[2]:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(filtered_df, x="math score", y="reading score", color="Risk Level", title="Scatter - Math vs Reading")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(filtered_df, x="math score", y="writing score", size="Average Score",
                         color="Risk Level", title="Bubble - Math vs Writing Performance")
        st.plotly_chart(fig, use_container_width=True)

# ================= TREEMAP / FUNNEL =================
with extra_tabs[3]:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.treemap(filtered_df, path=["gender", "Risk Level"], title="Treemap Gender > Risk Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        funnel_data = filtered_df["Risk Level"].value_counts().reset_index()
        funnel_data.columns = ["Risk Level", "Count"]
        fig = px.funnel(funnel_data, x="Count", y="Risk Level", title="Funnel Chart - Risk Funnel")
        st.plotly_chart(fig, use_container_width=True)

# ================= WATERFALL / RADAR =================
with extra_tabs[4]:
    col1, col2 = st.columns(2)

    with col1:
        avg_scores = {
            "Math": filtered_df["math score"].mean(),
            "Reading": filtered_df["reading score"].mean(),
            "Writing": filtered_df["writing score"].mean()
        }
        fig = go.Figure(go.Waterfall(
            x=list(avg_scores.keys()),
            y=list(avg_scores.values())
        ))
        fig.update_layout(title="Waterfall - Contribution of Avg Scores")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        radar_values = [
            filtered_df["math score"].mean(),
            filtered_df["reading score"].mean(),
            filtered_df["writing score"].mean()
        ]
        radar_labels = ["Math", "Reading", "Writing"]
        fig = go.Figure(data=go.Scatterpolar(r=radar_values, theta=radar_labels, fill="toself"))
        fig.update_layout(title="Radar Chart - Score Strength Comparison")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 🎯 PREDICTION FORM
# =====================================================
st.markdown("## 🎯 Predict Risk Level")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", df["gender"].unique())
        race = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
        parent_edu = st.selectbox("Parental Level of Education", df["parental level of education"].unique())
        lunch_type = st.selectbox("Lunch Type", df["lunch"].unique())
        prep_course = st.selectbox("Test Preparation Course", df["test preparation course"].unique())

    with col2:
        math = st.slider("Math Score", 0, 100, 60)
        reading = st.slider("Reading Score", 0, 100, 60)
        writing = st.slider("Writing Score", 0, 100, 60)
        avg = (math + reading + writing) / 3
        st.metric("Auto Average Score", f"{avg:.2f}")

    predict_btn = st.form_submit_button("🔍 Predict Risk")

# =====================================================
# 🤖 MODEL + RESULT
# =====================================================
if predict_btn:
    input_row = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch_type,
        "test preparation course": prep_course,
        "math score": math,
        "reading score": reading,
        "writing score": writing,
        "Average Score": avg
    }

    input_df = pd.DataFrame([input_row])

    for col in cat_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    X = df_encoded.drop(["Risk Level", "Risk Level Encoded"], axis=1)
    y = df_encoded["Risk Level Encoded"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction] * 100

    if prediction == 1:
        st.error(f"🚨 The student is predicted to be AT RISK\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"✅ The student is predicted to be NOT AT RISK\nConfidence: {confidence:.2f}%")
