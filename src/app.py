import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="Heart Disease Risk Dashboard", layout="wide")

# --- Configuration ---
# Default to repo-relative paths; override with env vars for flat EC2 deploys.
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "heart_disease_pipeline.pkl"))
DATA_PATH = os.environ.get("DATA_PATH", str(ROOT / "data" / "heart_disease_dataset.csv"))

DB_HOST = os.environ.get("DB_HOST", "")
DB_USER = os.environ.get("DB_USER", "admin")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", "heart_disease")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))

# --- Loaders ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)

pipeline = load_model()

# --- Feature labels and explainability helpers ---
FEATURE_LABELS = {
    "age": "Age", "sex": "Sex (Male=1)", "cp": "Chest Pain Type",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG", "thalach": "Max Heart Rate",
    "exang": "Exercise Angina", "oldpeak": "ST Depression",
    "slope": "Slope of Peak ST", "ca": "Major Vessels",
    "thal": "Thalassemia", "smoking": "Smoking",
    "diabetes": "Diabetes", "bmi": "BMI",
}

@st.cache_data
def get_coefficients():
    clf = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    features = list(FEATURE_LABELS.keys())
    return pd.DataFrame({
        "feature": features,
        "label": [FEATURE_LABELS[f] for f in features],
        "coefficient": clf.coef_[0],
        "odds_ratio": np.exp(clf.coef_[0]),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    })

def compute_contribution(input_df):
    scaled = pipeline.named_steps["scaler"].transform(input_df)[0]
    coef = pipeline.named_steps["classifier"].coef_[0]
    contribs = scaled * coef
    return pd.DataFrame({
        "feature": input_df.columns,
        "label": [FEATURE_LABELS[f] for f in input_df.columns],
        "input_value": input_df.iloc[0].values,
        "contribution": contribs,
    }).sort_values("contribution", key=abs, ascending=False)

# --- Database helpers ---
def get_db_connection():
    if not MYSQL_AVAILABLE or not DB_HOST:
        return None
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS,
        database=DB_NAME, port=DB_PORT, cursorclass=pymysql.cursors.DictCursor
    )

def save_prediction(features, prediction, probability):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO predictions
                   (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal, smoking, diabetes, bmi,
                    prediction, probability)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (*features, prediction, probability)
            )
        conn.commit()
    finally:
        conn.close()

@st.cache_data(ttl=60)
def get_all_predictions():
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
            rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    finally:
        conn.close()

# --- Main UI ---
st.title("Heart Disease Risk Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Calculator", "Population Insights", "Prediction Log", "Model Insights"]
)

# =====================================================================
# TAB 1: Risk Calculator
# =====================================================================
with tab1:
    st.subheader("Patient Risk Assessment")
    st.write("Enter patient information below to get a heart disease risk prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=55)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: {0: "Female", 1: "Male"}[x])
            cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: {
                1: "1 - Typical Angina", 2: "2 - Atypical Angina",
                3: "3 - Non-Anginal", 4: "4 - Asymptomatic"
            }[x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240)
            smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1],
                               format_func=lambda x: {0: "No", 1: "Yes"}[x])
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: {
                0: "0 - Normal", 1: "1 - ST-T Abnormality", 2: "2 - LV Hypertrophy"
            }[x])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                                 format_func=lambda x: {0: "No", 1: "Yes"}[x])
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

        with col3:
            slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3], format_func=lambda x: {
                1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"
            }[x])
            ca = st.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {
                3: "3 - Normal", 6: "6 - Fixed Defect", 7: "7 - Reversable Defect"
            }[x])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal, smoking, diabetes, bmi]
        columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                   "exang", "oldpeak", "slope", "ca", "thal", "smoking", "diabetes", "bmi"]

        input_df = pd.DataFrame([features], columns=columns)
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"**High Risk** of heart disease (probability: {probability:.1%})")
        else:
            st.success(f"**Low Risk** of heart disease (probability: {probability:.1%})")

        save_prediction(features, int(prediction), float(probability))
        st.cache_data.clear()  # refresh prediction log

        with st.expander("Why this score? (feature contributions)", expanded=True):
            breakdown = compute_contribution(input_df)
            breakdown["direction"] = breakdown["contribution"].apply(
                lambda x: "Raises risk" if x > 0 else "Lowers risk"
            )
            top = breakdown.head(10)
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    top, x="contribution", y="label", orientation="h",
                    color="direction",
                    color_discrete_map={"Raises risk": "#d62728", "Lowers risk": "#2ca02c"},
                    title="Top 10 factors driving this prediction",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top.set_index("label")["contribution"])
            st.caption(
                "Each bar shows how much that feature pushed this patient's risk up (red) "
                "or down (green), relative to an average patient. Longer bars = larger effect."
            )

# =====================================================================
# TAB 2: Population Insights
# =====================================================================
with tab2:
    st.subheader("Population Insights")
    st.write("Disease prevalence patterns across the training population (3,069 patients).")

    try:
        pop_df = load_training_data()
    except FileNotFoundError:
        st.error(f"Training dataset not found at {DATA_PATH}")
        st.stop()

    # KPI row
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Patients", f"{len(pop_df):,}")
    k2.metric("Disease Prevalence", f"{pop_df['heart_disease'].mean():.1%}")
    k3.metric("Average Age", f"{pop_df['age'].mean():.1f} years")

    st.divider()

    # Chart 1: Disease rate by age bucket
    st.markdown("#### Disease Rate by Age Group")
    age_bins = [20, 30, 40, 50, 60, 70, 100]
    age_labels = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    pop_df_chart = pop_df.copy()
    pop_df_chart["age_bucket"] = pd.cut(pop_df_chart["age"], bins=age_bins, labels=age_labels, right=False)
    age_rate = pop_df_chart.groupby("age_bucket", observed=True)["heart_disease"].mean().reset_index()
    age_rate.columns = ["Age Group", "Disease Rate"]

    if PLOTLY_AVAILABLE:
        fig = px.bar(age_rate, x="Age Group", y="Disease Rate", text_auto=".1%",
                     color="Disease Rate", color_continuous_scale="Reds")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(age_rate.set_index("Age Group"))
    st.caption("Disease prevalence rises with age, the strongest single predictor in this dataset.")

    st.divider()

    # Chart 2: Disease rate by chest pain type
    st.markdown("#### Disease Rate by Chest Pain Type")
    cp_labels = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"}
    cp_df = pop_df.copy()
    cp_df["Chest Pain Type"] = cp_df["cp"].map(cp_labels)
    cp_rate = cp_df.groupby("Chest Pain Type")["heart_disease"].mean().reset_index()
    cp_rate.columns = ["Chest Pain Type", "Disease Rate"]

    if PLOTLY_AVAILABLE:
        fig = px.bar(cp_rate, x="Chest Pain Type", y="Disease Rate", text_auto=".1%",
                     color="Disease Rate", color_continuous_scale="Reds")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(cp_rate.set_index("Chest Pain Type"))
    st.caption("Chest pain type is a clinically important discriminator of cardiac risk.")

    st.divider()

    # Chart 3: Disease rate by binary risk factor
    st.markdown("#### Disease Rate by Risk Factor")
    c1, c2, c3 = st.columns(3)
    for col_name, display_name, container, labels in [
        ("sex", "Sex", c1, {0: "Female", 1: "Male"}),
        ("smoking", "Smoking", c2, {0: "No", 1: "Yes"}),
        ("diabetes", "Diabetes", c3, {0: "No", 1: "Yes"}),
    ]:
        with container:
            tmp = pop_df.copy()
            tmp[display_name] = tmp[col_name].map(labels)
            rate = tmp.groupby(display_name)["heart_disease"].mean().reset_index()
            rate.columns = [display_name, "Disease Rate"]
            if PLOTLY_AVAILABLE:
                fig = px.bar(rate, x=display_name, y="Disease Rate", text_auto=".1%",
                             color="Disease Rate", color_continuous_scale="Reds")
                fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(rate.set_index(display_name))

    st.divider()

    # Chart 4: Cholesterol and max heart rate distributions
    st.markdown("#### Clinical Measurement Distributions")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Cholesterol by Disease Status**")
        chol_df = pop_df.copy()
        chol_df["Status"] = chol_df["heart_disease"].map({0: "No Disease", 1: "Disease"})
        if PLOTLY_AVAILABLE:
            fig = px.histogram(chol_df, x="chol", color="Status", barmode="overlay",
                               opacity=0.6, nbins=40, labels={"chol": "Cholesterol (mg/dL)"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(chol_df.groupby("Status")["chol"].describe())

    with c2:
        st.markdown("**Max Heart Rate by Disease Status**")
        thal_df = pop_df.copy()
        thal_df["Status"] = thal_df["heart_disease"].map({0: "No Disease", 1: "Disease"})
        if PLOTLY_AVAILABLE:
            fig = px.histogram(thal_df, x="thalach", color="Status", barmode="overlay",
                               opacity=0.6, nbins=40, labels={"thalach": "Max Heart Rate (bpm)"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(thal_df.groupby("Status")["thalach"].describe())

    st.caption("Overlapping distributions show how each measurement differs between disease and non-disease groups.")

# =====================================================================
# TAB 3: Prediction Log Dashboard
# =====================================================================
with tab3:
    st.subheader("Prediction Log")
    st.write("All patient screenings run through this system, with filters and export.")

    preds = get_all_predictions()
    if preds is None:
        st.info("Database not configured. Set DB_HOST, DB_PASS environment variables to connect to RDS.")
    elif preds.empty:
        st.info("No predictions recorded yet. Run a prediction on the Risk Calculator tab to get started.")
    else:
        # KPI row
        total = len(preds)
        high_risk_pct = (preds["prediction"] == 1).mean()
        avg_prob = preds["probability"].mean()

        if "created_at" in preds.columns:
            preds["created_at"] = pd.to_datetime(preds["created_at"])
            recent_24h = (preds["created_at"] >= datetime.now() - timedelta(hours=24)).sum()
        else:
            recent_24h = 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", f"{total:,}")
        k2.metric("High Risk Flagged", f"{high_risk_pct:.1%}")
        k3.metric("Avg Risk Score", f"{avg_prob:.1%}")
        k4.metric("Last 24 Hours", f"{recent_24h}")

        st.divider()

        # Filters
        st.markdown("#### Filters")
        f1, f2, f3 = st.columns(3)
        with f1:
            risk_filter = st.radio("Risk Level", ["All", "High Risk", "Low Risk"], horizontal=True)
        with f2:
            age_min, age_max = st.slider("Age Range", 20, 100, (20, 100))
        with f3:
            sex_filter = st.radio("Sex", ["All", "Female", "Male"], horizontal=True)

        # Apply filters
        filtered = preds.copy()
        if risk_filter == "High Risk":
            filtered = filtered[filtered["prediction"] == 1]
        elif risk_filter == "Low Risk":
            filtered = filtered[filtered["prediction"] == 0]
        filtered = filtered[(filtered["age"] >= age_min) & (filtered["age"] <= age_max)]
        if sex_filter == "Female":
            filtered = filtered[filtered["sex"] == 0]
        elif sex_filter == "Male":
            filtered = filtered[filtered["sex"] == 1]

        st.markdown(f"**Showing {len(filtered)} of {total} predictions**")

        # Display table with readable labels
        if not filtered.empty:
            display = filtered.copy()
            display_cols = ["id", "created_at", "age", "sex", "cp", "chol", "thalach",
                            "prediction", "probability"]
            display = display[[c for c in display_cols if c in display.columns]]
            display["prediction"] = display["prediction"].map({0: "Low Risk", 1: "High Risk"})
            display["sex"] = display["sex"].map({0: "Female", 1: "Male"})
            display["probability"] = display["probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display, use_container_width=True, hide_index=True)

            # CSV export
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered results as CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions match the current filters.")

# =====================================================================
# TAB 4: Model Insights
# =====================================================================
with tab4:
    st.subheader("Model Insights")
    st.write(
        "The model is a Logistic Regression trained on 16 clinical and demographic "
        "features. The charts below show which features drive its predictions, and "
        "how a single feature's value affects risk when other factors are held constant."
    )

    coef_df = get_coefficients()

    st.markdown("### Global Feature Importance")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scale = st.radio("Scale", ["Coefficient", "Odds Ratio"], horizontal=True)
    with c2:
        top_n = st.slider("Show top N features", 5, 16, 10)
    with c3:
        sort_mode = st.radio(
            "Sort by",
            ["Impact (absolute)", "Positive first", "Negative first"],
            horizontal=True,
        )

    value_col = "coefficient" if scale == "Coefficient" else "odds_ratio"
    df_global = coef_df.copy()
    if sort_mode == "Impact (absolute)":
        df_global = df_global.reindex(
            df_global[value_col].abs().sort_values(ascending=False).index
        )
    elif sort_mode == "Positive first":
        df_global = df_global.sort_values(value_col, ascending=False)
    else:
        df_global = df_global.sort_values(value_col, ascending=True)
    df_global = df_global.head(top_n)

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df_global, x=value_col, y="label", orientation="h",
            color=value_col, color_continuous_scale="RdBu_r",
            color_continuous_midpoint=(1.0 if scale == "Odds Ratio" else 0.0),
            labels={value_col: scale, "label": "Feature"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_global.set_index("label")[value_col])

    if scale == "Odds Ratio":
        st.caption(
            "Odds ratios above 1.0 raise disease risk, below 1.0 lower it. "
            "For example, 1.4 = 40% higher odds per standard-deviation increase."
        )
    else:
        st.caption(
            "Coefficients are on standardized inputs (z-scores). Positive = raises "
            "log-odds of disease, negative = lowers. Magnitudes are directly comparable."
        )

    st.divider()
    st.markdown("### What-If Sensitivity Analysis")
    st.write(
        "Pick a feature and see how predicted risk changes across its full range, "
        "while all other features are held at the dataset average."
    )

    try:
        pop_for_sweep = load_training_data()
    except FileNotFoundError:
        st.error(f"Training dataset not found at {DATA_PATH}")
        pop_for_sweep = None

    if pop_for_sweep is not None:
        s1, s2 = st.columns([1, 2])
        with s1:
            sweep_feature = st.selectbox(
                "Feature to vary",
                options=list(FEATURE_LABELS.keys()),
                format_func=lambda x: FEATURE_LABELS[x],
                index=0,
            )
            baseline = pop_for_sweep.drop(columns=["heart_disease"]).mean().to_dict()
            feat_min = float(pop_for_sweep[sweep_feature].min())
            feat_max = float(pop_for_sweep[sweep_feature].max())
            st.caption(
                f"Sweeping **{FEATURE_LABELS[sweep_feature]}** from {feat_min:.1f} "
                f"to {feat_max:.1f}. Other features held at dataset averages."
            )

        with s2:
            sweep_values = np.linspace(feat_min, feat_max, 50)
            sweep_df = pd.DataFrame([baseline] * 50)
            sweep_df[sweep_feature] = sweep_values
            sweep_df = sweep_df[list(FEATURE_LABELS.keys())]
            probs = pipeline.predict_proba(sweep_df)[:, 1]

            if PLOTLY_AVAILABLE:
                fig = px.line(
                    x=sweep_values, y=probs,
                    labels={
                        "x": FEATURE_LABELS[sweep_feature],
                        "y": "Predicted Risk Probability",
                    },
                    title=f"How risk changes with {FEATURE_LABELS[sweep_feature]}",
                )
                fig.update_traces(line=dict(width=3))
                fig.update_layout(
                    yaxis_tickformat=".0%", yaxis_range=[0, 1], height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_df = pd.DataFrame(
                    {FEATURE_LABELS[sweep_feature]: sweep_values, "Risk": probs}
                ).set_index(FEATURE_LABELS[sweep_feature])
                st.line_chart(chart_df)
