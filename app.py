# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# --------------------------
# Synthetic experiment (your model)
# --------------------------
def synthetic_experiment(temp, bf):
    temp_opt = 500
    temp_effect = 21.32681303 - 0.173742816 * temp + 0.000515144 * temp**2 - 6.60346E-07 * temp**3 + 3.09609E-10 * temp**4
    bf_percent = bf * 100.0
    bf_effect = np.where(bf <= 0.01, 0.08 * bf_percent, -0.04 * bf_percent + 0.12)
    strength_base = 35.0
    strength = strength_base * np.exp(bf_effect + temp_effect)
    co2 = np.where(bf <= 0.05, 3 + 0.732 * bf_percent - 0.084 * bf_percent**2,
                   4.56 + 1.44/5*(bf_percent-5))
    return strength, co2

# --------------------------
# Helper
# --------------------------
def normalize(arr):
    return (arr - np.min(arr)) / (np.ptp(arr) + 1e-9)

def expected_improvement(mu, sigma, f_best, xi=0.01):
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return ei

# --------------------------
# Generate synthetic data for initial model
# --------------------------
np.random.seed(42)
temp_range = (350, 675)
bf_range = (0, 0.10)

n_init = 60
X_init = np.column_stack([
    np.random.uniform(temp_range[0], temp_range[1], n_init),
    np.random.uniform(bf_range[0], bf_range[1], n_init)
])
y1_init, y2_init = synthetic_experiment(X_init[:, 0], X_init[:, 1])

kernel = ConstantKernel(1.0) * Matern(length_scale=[100, 0.01], nu=1.5) + WhiteKernel(noise_level=1.0)
gpr_strength = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_init, y1_init)
gpr_co2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_init, y2_init)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌱 Biochar Concrete Digital Twin")
st.markdown("""
Interactive simulator of **biochar-enhanced concrete** performance.  
Use the sliders to explore how **pyrolysis temperature** and **biochar fraction** affect:
- **Compressive strength**
- **CO₂ sequestration capacity**
- **Combined sustainability objective**
""")

temp = st.slider("🔥 Pyrolysis Temperature (°C)", 350, 675, 500, step=5)
bf_percent = st.slider("🌾 Biochar Fraction (% of binder)", 0.0, 10.0, 5.0, step=0.5)
w_strength = st.slider("⚖️ Weight on Strength (vs CO₂)", 0.0, 1.0, 0.7, 0.05)

bf = bf_percent / 100.0

# Prediction at selected point
mu_s, std_s = gpr_strength.predict(np.array([[temp, bf]]), return_std=True)
mu_c, std_c = gpr_co2.predict(np.array([[temp, bf]]), return_std=True)

st.subheader("📊 Predicted Properties")
col1, col2 = st.columns(2)
col1.metric("Compressive Strength (MPa)", f"{mu_s[0]:.2f}", f"±{std_s[0]:.2f}")
col2.metric("CO₂ Sequestered (synthetic units)", f"{mu_c[0]:.3f}", f"±{std_c[0]:.3f}")

# --------------------------
# Surface plots
# --------------------------
nt = 80
temps = np.linspace(temp_range[0], temp_range[1], nt)
bfs = np.linspace(bf_range[0], bf_range[1], nt)
T, B = np.meshgrid(temps, bfs)
grid = np.column_stack([T.ravel(), B.ravel()])

mu_surf_s, _ = gpr_strength.predict(grid, return_std=True)
mu_surf_c, _ = gpr_co2.predict(grid, return_std=True)

# Normalize and combine
mu_s_n = normalize(mu_surf_s)
mu_c_n = normalize(mu_surf_c)
combined = w_strength * mu_s_n + (1 - w_strength) * mu_c_n
combined_grid = combined.reshape(T.shape)

# Plot contours
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
cs0 = axs[0].contourf(T, B*100, mu_surf_s.reshape(T.shape), levels=20)
axs[0].scatter(temp, bf*100, c='red', s=60, edgecolor='k')
axs[0].set_title("Predicted Strength (MPa)")
axs[0].set_xlabel("Temp (°C)"); axs[0].set_ylabel("Biochar %")
fig.colorbar(cs0, ax=axs[0])

cs1 = axs[1].contourf(T, B*100, mu_surf_c.reshape(T.shape), levels=20)
axs[1].scatter(temp, bf*100, c='red', s=60, edgecolor='k')
axs[1].set_title("Predicted CO₂ (synthetic)")
axs[1].set_xlabel("Temp (°C)"); axs[1].set_ylabel("Biochar %")
fig.colorbar(cs1, ax=axs[1])

cs2 = axs[2].contourf(T, B*100, combined_grid, levels=20)
axs[2].scatter(temp, bf*100, c='red', s=60, edgecolor='k')
axs[2].set_title(f"Combined Objective (w_strength={w_strength:.2f})")
axs[2].set_xlabel("Temp (°C)"); axs[2].set_ylabel("Biochar %")
fig.colorbar(cs2, ax=axs[2])

plt.tight_layout()
st.pyplot(fig)

# Highlight suggestion
best_idx = np.argmax(combined)
best_point = grid[best_idx]
st.success(f"💡 **Suggested next optimal experiment:** {best_point[0]:.1f} °C, {best_point[1]*100:.2f}% biochar")
