import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

st.set_page_config(page_title="🧱 Biochar–Concrete Digital Twin", layout="wide")

# ==============================
# Synthetic model definition
# ==============================
def synthetic_experiment(temp, bf):
    bf = bf / 100  # convert % to fraction
    temp_effect = (
        21.32681303
        - 0.173742816 * temp
        + 0.000515144 * temp**2
        - 6.60346e-07 * temp**3
        + 3.09609e-10 * temp**4
    )
    bf_percent = bf * 100.0

    bf_effect = np.where(bf <= 0.01, 0.08 * bf_percent, -0.04 * bf_percent + 0.12)
    strength_base = 35.0
    strength = strength_base * np.exp(bf_effect + temp_effect)
    co2 = np.where(
        bf <= 0.05,
        3 + 0.732 * bf_percent - 0.084 * bf_percent**2,
        4.56 + 1.44 / 5 * (bf_percent - 5),
    )
    return strength, co2


# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("🔥 Pyrolysis Temperature (°C)", 350, 675, 500, step=5)
bf = st.sidebar.slider("🌿 Biochar Fraction (mass % of binder)", 0.0, 10.0, 5.0, step=0.1)
w_strength = st.sidebar.slider("Weight: Strength", 0.0, 1.0, 0.7)
w_co2 = 1 - w_strength

# ==============================
# Compute predictions
# ==============================
strength, co2 = synthetic_experiment(temp, bf)

col1, col2 = st.columns(2)
col1.metric("🧩 Predicted Compressive Strength (MPa)", f"{strength:.2f}")
col2.metric("🌍 CO₂ Sequestered (kg eq/m³)", f"{co2:.3f}")

st.divider()

# ==============================
# Build synthetic data grid
# ==============================
temps = np.linspace(350, 675, 60)
bfs = np.linspace(0, 10, 60)
T, B = np.meshgrid(temps, bfs)
strengths, co2s = synthetic_experiment(T, B)

# Normalize
def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    return (arr - np.min(arr)) / (np.ptp(arr) + 1e-9)

norm_strength = normalize(strengths)
norm_co2 = normalize(co2s)
combined = w_strength * norm_strength + w_co2 * norm_co2
best_idx = np.unravel_index(np.argmax(combined), combined.shape)
best_temp = temps[best_idx[1]]
best_bf = bfs[best_idx[0]]

st.success(f"🎯 Optimal Mix: {best_temp:.1f} °C | {best_bf:.2f}% Biochar")
# ==============================
# 2D Heatmaps
# ==============================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

im1 = ax[0].imshow(strengths, extent=[350, 675, 0, 10], origin="lower", aspect="auto", cmap="viridis")
ax[0].set_title("Compressive Strength (MPa)")
ax[0].set_xlabel("Temperature (°C)")
ax[0].set_ylabel("Biochar (%)")
plt.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(co2s, extent=[350, 675, 0, 10], origin="lower", aspect="auto", cmap="plasma")
ax[1].set_title("CO₂ Sequestered (kg eq/m³)")
ax[1].set_xlabel("Temperature (°C)")
ax[1].set_ylabel("Biochar (%)")
plt.colorbar(im2, ax=ax[1])

im3 = ax[2].imshow(combined, extent=[350, 675, 0, 10], origin="lower", aspect="auto", cmap="cividis")
ax[2].set_title("Combined Objective")
ax[2].set_xlabel("Temperature (°C)")
ax[2].set_ylabel("Biochar (%)")
plt.colorbar(im3, ax=ax[2])

st.pyplot(fig)

# ==============================
# 3D Visualization (Plotly)
# ==============================
st.subheader("🌍 3D Surface Explorer")

fig3d = go.Figure()

fig3d.add_trace(go.Surface(
    x=temps, y=bfs, z=strengths,
    colorscale="Viridis", name="Strength (MPa)", opacity=0.9
))

fig3d2= go.Figure()
fig3d2.add_trace(go.Surface(
    x=temps, y=bfs, z=co2s,
    colorscale="Plasma", name="CO₂ Sequestered", opacity=0.6
))

fig3d3= go.Figure()
fig3d3.add_trace(go.Surface(
    x=temps, y=bfs, z=combined * strengths.max(),
    colorscale="Cividis", name="Combined Objective", showscale=False, opacity=0.5
))
fig3d.update_layout(
    title="3D Surface — Strength",
    scene=dict(
        xaxis_title="Temperature (°C)",
        yaxis_title="Biochar (%)",
        zaxis_title="Value (scaled)"
    ),
    height=700
)
st.plotly_chart(fig3d, use_container_width=True)

fig3d2.update_layout(
    title="3D Surface — CO₂",
    scene=dict(
        xaxis_title="Temperature (°C)",
        yaxis_title="Biochar (%)",
        zaxis_title="Value (scaled)"
    ),
    height=700
)
st.plotly_chart(fig3d2, use_container_width=True)

fig3d3.update_layout(
    title="3D Surfaces — Combined Objective",
    scene=dict(
        xaxis_title="Temperature (°C)",
        yaxis_title="Biochar (%)",
        zaxis_title="Value (scaled)"
    ),
    height=700
)
st.plotly_chart(fig3d3, use_container_width=True)

# ==============================
# Gaussian Process (Uncertainty Demo)
# ==============================
st.subheader("🧠 Research Mode: Gaussian Process Uncertainty (Demo)")

# Generate random initial samples
np.random.seed(42)
X_init = np.column_stack([
    np.random.uniform(350, 675, 40),
    np.random.uniform(0, 0.10, 40)
])
y_strength, y_co2 = synthetic_experiment(X_init[:, 0], X_init[:, 1]*100)

# Fit Gaussian Processes
kernel = ConstantKernel(1.0) * Matern(length_scale=[100, 0.01], nu=1.5) + WhiteKernel(noise_level=1.0)
gpr_strength = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
gpr_strength.fit(X_init, y_strength)

grid = np.column_stack([T.ravel(), (B.ravel()/100)])
mu_s, std_s = gpr_strength.predict(grid, return_std=True)
mu_s = np.asarray(mu_s, dtype=float).reshape(-1)
std_s = np.asarray(std_s, dtype=float).reshape(-1)

# Reshape for plotting
mu_s = mu_s.reshape(T.shape)
std_s = std_s.reshape(T.shape)

# Plot uncertainty surface
fig_u = go.Figure()
fig_u.add_trace(go.Surface(
    x=temps, y=bfs, z=mu_s,
    colorscale="Viridis", name="Mean Strength"
))
fig_u.add_trace(go.Surface(
    x=temps, y=bfs, z=mu_s + std_s,
    colorscale="Oranges", name="Mean + σ", opacity=0.4
))
fig_u.add_trace(go.Surface(
    x=temps, y=bfs, z=mu_s - std_s,
    colorscale="Blues", name="Mean - σ", opacity=0.4
))
fig_u.update_layout(
    title="Predicted Strength Surface with Uncertainty ±σ",
    scene=dict(
        xaxis_title="Temperature (°C)",
        yaxis_title="Biochar (%)",
        zaxis_title="Strength (MPa)"
    ),
    height=700
)
st.plotly_chart(fig_u, use_container_width=True)

st.markdown("""
---
### 🔍 About this App
This interactive dashboard acts as a **Digital Twin** for optimizing concrete with biochar.  
It combines:
- Synthetic models from literature-based equations  
- Gaussian Process regression (AI learning)
- Multi-objective weighting between strength & CO₂ sequestration  
- Real-time 2D and 3D visualization  

Future extensions can include:
- Bayesian optimization loops  
- Pareto front visualizations  
- Real experimental data integration
""")

