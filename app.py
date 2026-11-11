from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path(__file__).parent / "bike_demand_model.joblib"

st.set_page_config(
    page_title="BikeShare | Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { max-width: 1280px; padding-top: 1.7rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
.hero {
  background: linear-gradient(120deg, #102a43 0%, #155e75 58%, #0f766e 100%);
  padding: 30px 34px; border-radius: 22px; color: white; margin-bottom: 22px;
  box-shadow: 0 12px 28px rgba(16,42,67,.13);
}
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.2rem; margin: 0; }
.hero p { color: #d8edf1; margin: 7px 0 0 0; font-size: 1rem; }
.eyebrow { color: #a9d8dc; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
.section-card {
  background: white; border: 1px solid #e6edf4; border-radius: 18px;
  padding: 21px 23px; box-shadow: 0 4px 16px rgba(16,42,67,.035);
}
div[data-testid="stForm"] {
  background: white; border: 1px solid #e6edf4; border-radius: 18px;
  padding: 20px 22px;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  background: #087e8b; color: white; border: 0; border-radius: 10px;
  min-height: 2.8rem; font-weight: 700;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  background: #066873; color: white;
}
[data-testid="stMetric"] {
  background: white; border: 1px solid #e6edf4; padding: 17px 19px;
  border-radius: 15px;
}
[data-testid="stMetricLabel"] { color: #627d98; }
[data-testid="stMetricValue"] { color: #102a43; }
h2, h3 { font-family: 'Manrope', sans-serif; color: #102a43; }
div[data-testid="stSidebar"] { background: #102a43; }
div[data-testid="stSidebar"] * { color: #edf6f9 !important; }
div[data-testid="stSidebar"] .stCaption { color: #bdd3dc !important; }
hr { border-color: #e6edf4; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Demand intelligence · hourly forecasting</div>
  <h1>🚲 BikeShare</h1>
  <p>Understand expected bike rental demand from time, calendar and weather conditions.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("The trained model file is missing.")
    st.markdown(
        "Add **`bike_demand_model.joblib`** to the same GitHub repository folder as `app.py`. "
        "Export it by running the model-saving cell in your notebook."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load the model artifact: {exc}")
    st.stop()

# Read the exact category levels learned by the notebook's fitted OneHotEncoder.
try:
    preprocessor = model.named_steps["preprocess"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_columns = list(preprocessor.transformers_[1][2])
    learned_categories = {
        col: [v for v in vals if pd.notna(v)]
        for col, vals in zip(categorical_columns, encoder.categories_)
    }
except Exception:
    st.error("The uploaded artifact does not look like the complete fitted notebook pipeline.")
    st.stop()

with st.sidebar:
    st.markdown("## Project menu")
    st.markdown("**Demand prediction**")
    st.caption("Model: HistGradientBoostingRegressor")
    st.divider()
    st.markdown("### About this model")
    st.caption(
        "The notebook saves a Scikit-learn pipeline containing the preprocessing "
        "transformer and trained regression model."
    )
    st.markdown("### Input guidance")
    st.caption("Temperature, feels-like temperature, humidity and wind speed use the dataset's normalized 0–1 scale.")
    st.divider()
    st.caption("BikeShare · Academic project")

st.markdown("## Create a demand estimate")
st.caption("Choose the conditions for the hour you want to forecast.")

with st.form("prediction_form"):
    date_col, time_col = st.columns([1.2, 1])
    with date_col:
        selected_date = st.date_input(
            "Date",
            value=date(2012, 9, 12),
            min_value=date(2011, 1, 1),
            max_value=date(2012, 12, 31),
        )
    with time_col:
        hour = st.slider("Hour of day", min_value=0, max_value=23, value=17, format="%d:00")

    st.markdown("#### Calendar conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_options = learned_categories.get("season", ["spring", "summer", "fall", "winter"])
        season = st.selectbox("Season", season_options)
    with c2:
        holiday_options = learned_categories.get("holiday", ["No", "Yes"])
        holiday = st.selectbox("Public holiday", holiday_options)
    with c3:
        working_options = learned_categories.get("workingday", ["Working Day", "Non-Working Day"])
        workingday = st.selectbox("Working day", working_options)

    st.markdown("#### Weather conditions")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        weather_options = learned_categories.get("weathersit", ["Clear", "Mist", "Light Snow/Rain", "Heavy Rain/Snow"])
        weather = st.selectbox("Weather", weather_options)
    with w2:
        temp = st.slider("Temperature", 0.0, 1.0, 0.65, 0.01)
    with w3:
        atemp = st.slider("Feels-like temperature", 0.0, 1.0, 0.62, 0.01)
    with w4:
        humidity = st.slider("Humidity", 0.0, 1.0, 0.45, 0.01)
    windspeed = st.slider("Wind speed", 0.0, 1.0, 0.15, 0.01)

    submitted = st.form_submit_button("Generate demand prediction", use_container_width=True)

st.markdown("---")
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown("### Prediction workspace")
    st.markdown(
        '<div class="section-card"><b>What this tool does</b><br>'
        '<span style="color:#627d98">The model estimates the total number of bike rentals for one selected hour. '
        'It uses the same feature columns and preprocessing pipeline prepared during model training.</span></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown("### Model snapshot")
    st.metric("Selected algorithm", "HistGradientBoosting")
    st.metric("Prediction target", "Hourly rentals · cnt")
    st.caption("Model evaluation in the notebook: test MAE 49.83 · RMSE 73.78 · R² 88.24%. These are notebook-reported metrics.")

if submitted:
    # The dataset's weekday convention is Sunday=0 through Saturday=6.
    weekday = (selected_date.weekday() + 1) % 7
    month = selected_date.month
    yr = float(selected_date.year)
    dayofyear = selected_date.timetuple().tm_yday

    # These are the exact 20 predictor columns constructed in notebook cell 8/15.
    row = {
        "season": season,
        "yr": yr,
        "mnth": float(month),
        "hr": int(hour),
        "holiday": holiday,
        "weekday": int(weekday),
        "workingday": workingday,
        "weathersit": weather,
        "temp": float(temp),
        "atemp": float(atemp),
        "hum": float(humidity),
        "windspeed": float(windspeed),
        "year": int(selected_date.year),
        "day": int(selected_date.day),
        "dayofyear": int(dayofyear),
        "month": int(month),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
    }

    try:
        input_df = pd.DataFrame([row])
        prediction = max(0, int(np.rint(float(model.predict(input_df)[0]))))

        st.markdown("---")
        st.markdown("## Your prediction")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Estimated rentals", f"{prediction:,} bikes")
        metric_cols[1].metric("Forecast hour", f"{hour:02d}:00")
        metric_cols[2].metric("Forecast date", selected_date.strftime("%d %b %Y"))

        # A single-bar visual reinforces the point forecast without implying a time series.
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"valueformat": ",.0f", "suffix": " bikes"},
            title={"text": "Predicted rentals for the selected hour"},
            gauge={
                "axis": {"range": [0, max(100, int(prediction * 1.25) + 1)]},
                "bar": {"color": "#087e8b"},
                "bgcolor": "#edf3f7",
                "borderwidth": 0,
            },
        ))
        fig.update_layout(height=270, margin=dict(l=25, r=25, t=65, b=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Inspect the exact model input"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the model file was exported from the same notebook and that "
            "its preprocessing pipeline is included."
        )
else:
    st.info("Select your conditions above, then click **Generate demand prediction**.")

st.markdown("---")
st.caption(
    "BikeShare Demand Intelligence · Predictions are estimates, not guaranteed rental counts. "
    "Model trained on historical 2011–2012 data."
)
