import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Microalgae Lipid Productivity Predictor",
    layout="centered",
    page_icon="🌱"
)

# Styling
st.markdown(
    """
    <style>
    body { background-color: #0A0E1A; color: white; font-family: Arial, sans-serif; }
    .sidebar { background: #1B2435; }
    .stButton>button { background: #2B73B5; color: white; padding: 0.7rem; border-radius: 10px; }
    .stApp { background-color: #0A0E1A; }
    h1 { color: #63B3ED; }
    h2,h3,h4 { color: #63B3ED; }
    footer { visibility: hidden; }
    footer:after {
        content: 'App designed by Aditya Atul';
        visibility: visible;
        display: block;
        padding: 1rem;
        color: #A0AEC0;
        text-align: center;
        position: relative;
        top: -20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 Microalgae Lipid Productivity Predictor")

st.sidebar.header("Input Parameters ⚙️")

uploaded_file = st.sidebar.file_uploader(
    "Upload your microalgae CSV file",
    type=["csv"],
    help="CSV file must contain columns like Light (µmol photons/m²/s), pH, Nitrate (mg/L), etc."
)

@st.cache_data
def load_and_train(file):
    lipid_db = pd.read_csv(file)
    encoding = pd.get_dummies(lipid_db, columns=['Strain'], drop_first=False)
    X = encoding.drop(columns=['Lipid Productivity (g/L/day)'])
    y = encoding['Lipid Productivity (g/L/day)']
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    return lipid_db, X, y, model, rmse, r2

if uploaded_file:
    lipid_db, X, y, model, rmse, r2 = load_and_train(uploaded_file)
    strain_cols = [c for c in X.columns if c.startswith('Strain_')]

    Light = st.sidebar.number_input('Light (µmol photons/m²/s)', min_value=0.0, value=100.0)
    pH = st.sidebar.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
    Salinity = st.sidebar.number_input('Salinity (PSU)', min_value=0.0, value=10.0)
    Temperature = st.sidebar.number_input('Temperature (°C)', min_value=0.0, value=25.0)
    N_P_Ratio = st.sidebar.number_input('N:P Ratio', min_value=0.0, value=10.0)
    Nitrate = st.sidebar.number_input('Nitrate (mg/L)', min_value=0.0, value=50.0)
    Phosphate = st.sidebar.number_input('Phosphate (mg/L)', min_value=0.0, value=5.0)
    Lipid_Percentage = st.sidebar.number_input('Lipid Percentage (%)', min_value=0.0, value=20.0)
    Biomass = st.sidebar.number_input('Biomass (g/L)', min_value=0.0, value=1.0)
    Culture_time_days = st.sidebar.number_input('Culture time (days)', min_value=0.0, value=10.0)

    strain = st.sidebar.selectbox(
        "Strain",
        [
            'Chlamydomonas reinhardtii',
            'Chlorella sorokiniana',
            'Chlorella vulgaris',
            'Monoraphidium braunii',
            'Nannochloropsis gaditana'
        ]
    )

    strain_vec = [0]*len(strain_cols)
    if f'Strain_{strain}' in strain_cols:
        strain_vec[strain_cols.index(f'Strain_{strain}')] = 1

    vals = np.array(
        [Light, pH, Salinity, Temperature, N_P_Ratio, Nitrate, Phosphate,
         Lipid_Percentage, Biomass, Culture_time_days] + strain_vec
    ).reshape(1, -1)

    pred = model.predict(vals)[0]

    st.markdown(
        f"""
        **Predicted Lipid Productivity:**  
        <h2>{pred:.4f} g/L/day</h2>

        **Model Performance:**  
        • RMSE: {rmse:.4f}  
        • R²: {r2:.4f}
        """,
        unsafe_allow_html=True
    )

    st.subheader("Quick Data Overview")
    fig = px.scatter(
        lipid_db,
        x='Light (µmol photons/m²/s)',
        y='Lipid Productivity (g/L/day)',
        color='Strain',
        opacity=0.7,
        title='Light vs Lipid Productivity'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(" Please upload your CSV file to use this application!")