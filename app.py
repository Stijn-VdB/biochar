import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

st.set_page_config(page_title="🧱 Biochar–Concrete Digital Twin 2.0", layout="wide")

# --- Functions ---
def synthetic_experiment(temp, bf):
    """Synthetic behavior based on literature-inspired equations."""
    bf = bf / 100
    temp_effect = 21.32681303 - 0.173742816*temp + 0.000515144*temp**2 - 6.60346E-07*temp**3 + 3.09609E-10*temp**4
    bf_percent = bf * 100.0
    bf_effect = 0.08 * bf_percent if bf <= 0.01 else -0.04 * bf_percent + 0.12
    strength_base = 35.0
    strength = strength_base * np.exp(bf_effect + temp_effect)
    co2 = 3 + 0.732*bf_percent - 0.084*bf_percent**2 if bf <= 0.05 else 4.56 + 1.44/5*(bf_percent-5)
    return strength, co2

# --- Input ranges ---
temp_range = (350, 675)
bf_range = (0, 10)
temps = np.linspace(*temp_range, 50)
bfs = np.linspace(*bf_range, 50)
T, B = np.meshgrid(temps, bfs)

# --- Generate synthetic data for GP training ---
np.random.seed(42)
X_init = np.column_stack([
    np.random.uniform(temp_range[0], temp_range[1], 60),
    np.random.uniform(bf_range[0], bf_range[1], 60)
])
y_strength, y_co2 = synthetic_experiment(X_init[:,0], X_init[:,1])

kernel = ConstantKernel(1.0) * Matern(length_scale=[100, 0.01], nu=1.5) + WhiteKernel(noise_level=1.0)
gp_strength = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp_co2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp_strength.fit(X_init, y_strength)
gp_co2.fit(X_init, y_co2)

# --- Page Layout ---
tabs = st.tabs(["🌍 Simulation Dashboard", "🌋 3D Surface Explorer", "🤖 AI / Uncertainty", "📊 Experiment Planner"])

# --- Tab 1: Simulation Dashboard ---
with tabs[0]:
    st.header("🌍 Biochar–Concrete Simulation")
    temp = st.slider("Pyrolysis Temperature (°C)", 350, 675, 500, step=5)
    bf = st.slider("Biochar Fraction (%)", 0.0, 10.0, 5.0, step=0.1)
    s, c = synthetic_experiment(temp, bf)
    st.metric("🧩 Strength (MPa)", f"{s:.2f}")
    st.metric("🌱 CO₂ Sequestered (kg eq/m³)", f"{c:.2f}")

# --- Tab 2: 3D Surface Explorer ---
with tabs[1]:
    st.header("🌋 3D Surface Visualization")
    S, C = np.zeros_like(T), np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            s, c = synthetic_experiment(T[i,j], B[i,j])
            S[i,j], C[i,j] = s, c

    fig = go.Figure()
    fig.add_trace(go.Surface(x=T, y=B, z=S, colorscale="Viridis", name="Strength"))
    fig.add_trace(go.Surface(x=T, y=B, z=C, colorscale="Plasma", showscale=False, opacity=0.7, name="CO2"))
    fig.update_layout(scene=dict(
        xaxis_title='Temp (°C)', yaxis_title='Biochar (%)', zaxis_title='Value'
    ), height=700)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: AI / Uncertainty ---
with tabs[2]:
    st.header("🤖 Gaussian Process Predictions & Uncertainty")
    grid = np.column_stack([T.ravel(), B.ravel()])
    mu_s, std_s = gp_strength.predict(grid, return_std=True)
    mu_c, std_c = gp_co2.predict(grid, return_std=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Surface(x=T, y=B, z=std_s.reshape(T.shape), colorscale="Inferno"))
    fig2.update_layout(scene=dict(xaxis_title='Temp (°C)', yaxis_title='Biochar (%)', zaxis_title='Uncertainty (σ)'),
                       title="Prediction Uncertainty (Strength GP)", height=700)
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 4: Experiment Planner ---
with tabs[3]:
    st.header("📊 Active Learning / Next Experiment Suggestion")
    mu_comb = 0.7 * (mu_s - mu_s.min()) / (mu_s.ptp() + 1e-9) + 0.3 * (mu_c - mu_c.min()) / (mu_c.ptp() + 1e-9)
    var_comb = 0.7**2 * std_s**2 + 0.3**2 * std_c**2
    std_comb = np.sqrt(var_comb)
    f_best = np.max(mu_comb)
    Z = (mu_comb - f_best) / std_comb
    ei = (mu_comb - f_best) * norm.cdf(Z) + std_comb * norm.pdf(Z)
    ei[std_comb <= 1e-9] = 0
    best_idx = np.argmax(ei)
    best_T, best_B = grid[best_idx]

    st.success(f"🎯 Suggested Next Experiment: Temp = {best_T:.1f} °C | Biochar = {best_B:.2f}%")

    fig3 = go.Figure(data=[go.Surface(x=T, y=B, z=ei.reshape(T.shape), colorscale="Turbo")])
    fig3.update_layout(scene=dict(xaxis_title='Temp (°C)', yaxis_title='Biochar (%)', zaxis_title='Expected Improvement'),
                       title="Acquisition Surface (EI)", height=700)
    st.plotly_chart(fig3, use_container_width=True)
