import os
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
lipid_db = pd.read_csv('microalage.csv')  # Make sure this file is in the same folder
encoding = pd.get_dummies(lipid_db, columns=['Strain'], drop_first=False)

X = encoding.drop(columns=['Lipid Productivity (g/L/day)'])
y = encoding['Lipid Productivity (g/L/day)']

# Fit model
model = LinearRegression()
model.fit(X, y)
Lipid_prediction = model.predict(X)

# Metrics
mse = mean_squared_error(y, Lipid_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y, Lipid_prediction)

# Helper to predict lipid productivity
def predict_lipid(
    Light, pH, Salinity, Temperature, N_P_Ratio, Nitrate, Phosphate,
    Lipid_Percentage, Biomass, Culture_time_days, strain
):
    strain_vec = [0]*len([c for c in X.columns if c.startswith('Strain_')])
    strains = [c for c in X.columns if c.startswith('Strain_')]
    if f'Strain_{strain}' in strains:
        idx = strains.index(f'Strain_{strain}')
        strain_vec[idx] = 1

    vals = [
        Light, pH, Salinity, Temperature, N_P_Ratio, Nitrate, Phosphate,
        Lipid_Percentage, Biomass, Culture_time_days
    ] + strain_vec
    vals = np.array(vals).reshape(1, -1)
    pred = model.predict(vals)[0]
    return f"{pred:.4f} g/L/day"

# Build interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌱 Microalgae Lipid Productivity Predictor  
        **App designed and developed by Aditya Atul**
        ---
        """
    )
    with gr.Row():
        Light = gr.Number(label='Light (µmol photons/m²/s)')
        pH = gr.Number(label='pH')
        Salinity = gr.Number(label='Salinity (PSU)')
    with gr.Row():
        Temperature = gr.Number(label='Temperature (°C)')
        N_P_Ratio = gr.Number(label='N:P Ratio')
        Nitrate = gr.Number(label='Nitrate (mg/L)')
    with gr.Row():
        Phosphate = gr.Number(label='Phosphate (mg/L)')
        Lipid_Percentage = gr.Number(label='Lipid Percentage (%)')
        Biomass = gr.Number(label='Biomass (g/L)')
    with gr.Row():
        Culture_time_days = gr.Number(label='Culture time (days)')
        strain = gr.Dropdown(
            label='Strain',
            choices=[
                'Chlamydomonas reinhardtii',
                'Chlorella sorokiniana',
                'Chlorella vulgaris',
                'Monoraphidium braunii',
                'Nannochloropsis gaditana'
            ]
        )
    output = gr.Textbox(label='Predicted Lipid Productivity')
    gr.Markdown(f"*Model RMSE: {rmse:.4f} • R²: {r2:.4f}*")

    btn = gr.Button("Predict 🚀")
    btn.click(
        predict_lipid,
        inputs=[Light, pH, Salinity, Temperature, N_P_Ratio, Nitrate,
                 Phosphate, Lipid_Percentage, Biomass, Culture_time_days, strain],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        *App created by Aditya Atul • Powered by Gradio*
        """
    )

# Launch the app for Render
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
