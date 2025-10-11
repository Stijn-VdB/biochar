tab1, tab2 = st.tabs(["Main Dashboard", "Multiscale Visualization"])
with tab1:
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






with tab2:
        """
    Streamlit app: Multi-scale façade panel visualization (macro -> meso -> micro)
    
    How to run
    ----------
    1. Install dependencies: pip install streamlit plotly numpy pandas scipy
    2. Run: streamlit run streamlit_multiscale_microstructure.py
    
    What this app does (high level)
    -------------------------------
    - Lets you change a concrete mix (biochar %, w/c ratio, aggregate %) with sliders.
    - Computes *placeholder* material properties (porosity, CO2 uptake, estimated strength) using fictional functions you must replace with literature/experimental models.
    - Visualizes three scales:
        * Macro: the façade panel as a flat panel with a CO2 / biochar color map.
        * Meso: a small cube cutout showing a 3D grid where color and marker size indicate local porosity.
        * Micro: a procedurally generated particle cloud (biochar particles + pores) inside a small cubic volume.
    - Provides CSV download of the generated microstructure point cloud for further processing or import into Blender.
    
    IMPORTANT: This code uses *representative* and *visual* models, not a physically exact micromechanical simulation.
    Everyplace the code uses an assumed numeric formula or constant is marked with an "ASSUMPTION:" comment and shown in the UI under "Assumptions & checks".
    
    You should replace the placeholder functions (estimate_porosity, estimate_co2_uptake, estimate_strength) with formulas from literature or your experiments.
    
    """
    
    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import time
    
    # -------------------------
    # ------------ UI / Layout
    # -------------------------
    st.set_page_config(layout="wide", page_title="Multiscale Biochar–Concrete Viewer")
    st.title("Multiscale Façade Panel — Macro → Meso → Micro")
    
    # Sidebar inputs (composition)
    st.sidebar.header("Mixture composition (editable)")
    biochar_pct = st.sidebar.slider("Biochar content (% by mass of binder)", 0.0, 20.0, 5.0, 0.5)
    wc_ratio = st.sidebar.slider("Water/Cement ratio (w/c)", 0.25, 0.70, 0.45, 0.01)
    aggregate_pct = st.sidebar.slider("Coarse aggregate (% of mix mass)", 0.0, 70.0, 40.0, 1.0)
    
    st.sidebar.header("Geometry / Visualization")
    panel_w = st.sidebar.number_input("Panel width (m)", 0.2, 5.0, 1.0, 0.1)
    panel_h = st.sidebar.number_input("Panel height (m)", 0.2, 5.0, 2.0, 0.1)
    micro_cube_mm = st.sidebar.slider("Micro cube size (mm)", 1, 50, 10)
    micro_particle_count = st.sidebar.slider("Micro particle count (visual)", 100, 5000, 800, step=50)
    
    scale_choice = st.sidebar.selectbox("Scale to view", ("Macro (panel)", "Meso (cutout)", "Micro (zoom)", "Play zoom animation"))
    
    st.sidebar.header("Export / Debug")
    export_csv = st.sidebar.checkbox("Offer microstructure CSV download", True)
    
    # -------------------------
    # ---- Placeholder models
    # -------------------------
    # NOTE: these are *fictional* placeholder functions. Replace with literature-backed formulas.
    
    def estimate_porosity(biochar_pct, wc_ratio, aggregate_pct):
        """
        Estimate bulk porosity (volume fraction) from composition.
    
        ASSUMPTION: linear model. Replace with experimental correlation.
        - base_porosity: typical for dense concrete without biochar
        - effect of biochar: increases porosity per % of biochar
        - effect of w/c: higher w/c increases porosity
        - effect of aggregate: more aggregate lowers paste volume -> lower porosity
        """
        base_porosity = 0.08  # ASSUMPTION: 8% base porosity for dense concrete
        porosity = base_porosity + 0.015 * biochar_pct + 0.08 * (wc_ratio - 0.40) - 0.001 * aggregate_pct
        porosity = np.clip(porosity, 0.005, 0.60)
        return porosity
    
    
    def estimate_co2_uptake_per_m3(biochar_pct, porosity):
        """
        Estimate CO2 uptake (kg CO2 per m^3 of concrete) due to biochar and accessible pore surface.
    
        ASSUMPTION: simplified model:
          CO2 uptake ~= biochar_mass_per_m3 * adsorption_capacity * accessibility_factor
        - concrete density assumed 2400 kg/m3
        - adsorption_capacity: assumed kgCO2 per kg biochar (fictional)
        - accessibility_factor: function of porosity (higher porosity => more accessible surface)
        Replace with your adsorption isotherms / BET data.
        """
        density = 2400.0  # ASSUMPTION: concrete bulk density kg/m3
        biochar_mass_per_m3 = density * (biochar_pct / 100.0)
        adsorption_capacity = 0.25  # ASSUMPTION: kg CO2 captured per kg biochar (replace from literature)
        accessibility = 0.5 + 0.9 * porosity  # ASSUMPTION: arbitrary mapping 0.5..1.4
        uptake = biochar_mass_per_m3 * adsorption_capacity * accessibility
        return uptake
    
    
    def estimate_strength(biochar_pct, porosity, wc_ratio):
        """
        Estimate 28-day compressive strength (MPa) as a function of mix.
    
        ASSUMPTION: simplified inverse relation with porosity and w/c; biochar acts as weak filler beyond a small fraction.
        Replace with experimental regression.
        """
        base_strength = 50.0  # ASSUMPTION: MPa for reference mix
        strength = base_strength * (1 - (porosity - 0.08)) * (0.45 / wc_ratio)
        # penalize extreme biochar content (above 8%) more strongly
        if biochar_pct > 8:
            strength *= max(0.5, 1.0 - (biochar_pct - 8) * 0.05)
        return max(5.0, strength)
    
    # -------------------------
    # ---- Visualization tools
    # -------------------------
    
    def create_macro_figure(panel_w, panel_h, co2_uptake, biochar_pct):
        """Create a simple rectangular panel mesh colored by CO2 uptake / biochar content."""
        x = np.array([0, panel_w, panel_w, 0])
        y = np.array([0, 0, panel_h, panel_h])
        z = np.zeros(4)
    
        # single face mesh (two triangles)
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]
    
        # color - map co2_uptake to a color intensity (fictional normalization)
        intensity = np.repeat(co2_uptake, 4)
    
        mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensity,
                         colorscale='Viridis', showscale=True, opacity=0.9)
    
        # annotation text
        text = f"Biochar {biochar_pct:.1f}%  •  CO₂ uptake ~ {co2_uptake:.1f} kg/m³"
    
        fig = go.Figure(data=[mesh])
        fig.update_layout(scene=dict(aspectmode='auto',
                                     xaxis=dict(visible=False),
                                     yaxis=dict(visible=False),
                                     zaxis=dict(visible=False)),
                          margin=dict(l=0, r=0, t=30, b=0),
                          title=text)
        camera = dict(eye=dict(x=1.5, y=1.5, z=0.7))
        fig.update_layout(scene_camera=camera)
        return fig
    
    
    def create_meso_figure(cube_size_mm, porosity, grid_n=16):
        """
        Create a meso-scale cube visualization: a regular grid of points colored by local porosity.
        We simulate heterogeneity using random noise + a depth gradient.
        """
        # convert mm to meters for consistent units in display (not critical)
        s = cube_size_mm / 1000.0
    
        # create a grid of points
        lin = np.linspace(-s/2, s/2, grid_n)
        X, Y, Z = np.meshgrid(lin, lin, lin)
        Xf = X.flatten(); Yf = Y.flatten(); Zf = Z.flatten()
    
        # create a synthetic porosity field: base value + depth gradient + random noise
        depth_gradient = (Zf - Zf.min()) / (Zf.max() - Zf.min())  # 0..1
        noise = np.random.normal(scale=0.02, size=Xf.shape)
        por_field = np.clip(porosity * (0.9 + 0.4 * depth_gradient) + noise, 0.001, 0.6)
    
        marker_size = np.interp(por_field, [por_field.min(), por_field.max()], [4, 10])
    
        scatter = go.Scatter3d(x=Xf, y=Yf, z=Zf, mode='markers',
                               marker=dict(size=marker_size, color=por_field, colorscale='Inferno', showscale=True,
                                           colorbar=dict(title='Porosity')))
    
        # boundary box mesh for the cube
        # 8 vertices of cube for mesh box (visualization only)
        c = s/2
        verts = np.array([[-c, -c, -c], [c, -c, -c], [c, c, -c], [-c, c, -c],
                          [-c, -c, c], [c, -c, c], [c, c, c], [-c, c, c]])
        x, y, z = verts.T
        # 12 triangles to show edges - we'll use go.Mesh3d with low opacity
        mesh = go.Mesh3d(x=x, y=y, z=z, i=[0,0,0,4,4,4,0,1,2,4,5,6],
                         j=[1,2,3,5,6,7,4,2,3,5,6,7],
                         k=[2,3,0,6,7,4,1,3,0,6,7,4],
                         color='lightgray', opacity=0.05, showscale=False)
    
        fig = go.Figure(data=[mesh, scatter])
        fig.update_layout(scene=dict(aspectmode='cube', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                          margin=dict(l=0, r=0, t=30, b=0), title=f"Meso cube: {cube_size_mm} mm — representative porosity ~ {porosity:.3f}")
        camera = dict(eye=dict(x=1.2, y=1.2, z=1.2))
        fig.update_layout(scene_camera=camera)
        return fig
    
    
    def create_micro_figure(cube_size_mm, biochar_pct, porosity, n_particles=800):
        """
        Create a micro-scale visualization as a point cloud of "biochar particles" and "pores".
        - We represent particles as scatter markers sized by particle radius.
        - This is a *visual* representation (spheres approximated by markers), not a mesh packing.
    
        ASSUMPTIONS:
          - Particles are spheres (biochar and pores) with lognormal size distribution
          - Number of biochar particles is proportional to biochar_pct
          - Pore number is proportional to porosity
          - Overlaps are allowed (we do not perform collision detection / packing)
        Replace with a proper packing algorithm or CT-derived geometry for higher fidelity.
        """
        # volume of cube in m^3
        cube_m = (cube_size_mm / 1000.0) ** 3
    
        # particle counts (visual heuristic, not physical)
        # ASSUMPTION: fraction of particles that are biochar depends on biochar_pct
        frac_bio = np.clip(biochar_pct / 20.0, 0.01, 0.9)
        n_bio = int(n_particles * frac_bio)
        n_pores = n_particles - n_bio
    
        # sample positions uniformly inside cube [-s/2, s/2]
        s = cube_size_mm / 1000.0
        def rand_points(n):
            return np.random.uniform(-s/2, s/2, size=(n, 3))
    
        bio_pts = rand_points(n_bio)
        pore_pts = rand_points(n_pores)
    
        # particle sizes (mm) using a lognormal distribution
        # ASSUMPTION: biochar particle median size 0.2 mm, pores median size 0.05 mm
        bio_r_mm = np.random.lognormal(mean=np.log(0.2), sigma=0.6, size=n_bio)
        pore_r_mm = np.random.lognormal(mean=np.log(0.05), sigma=0.8, size=n_pores)
    
        # convert sizes to marker pixels for Plotly (visual scaling)
        # ASSUMPTION: visual scale factor
        scale_factor = 40.0
        bio_marker_size = np.clip(bio_r_mm * scale_factor, 2, 40)
        pore_marker_size = np.clip(pore_r_mm * scale_factor, 1, 30)
    
        trace_bio = go.Scatter3d(x=bio_pts[:,0], y=bio_pts[:,1], z=bio_pts[:,2], mode='markers',
                                 marker=dict(size=bio_marker_size, color='black', opacity=0.9), name='Biochar particles')
        trace_pores = go.Scatter3d(x=pore_pts[:,0], y=pore_pts[:,1], z=pore_pts[:,2], mode='markers',
                                   marker=dict(size=pore_marker_size, color='lightblue', opacity=0.6), name='Pores')
    
        # cube boundary
        c = s/2
        verts = np.array([[-c, -c, -c], [c, -c, -c], [c, c, -c], [-c, c, -c],
                          [-c, -c, c], [c, -c, c], [c, c, c], [-c, c, c]])
        x, y, z = verts.T
        mesh = go.Mesh3d(x=x, y=y, z=z, i=[0,0,0,4,4,4,0,1,2,4,5,6],
                         j=[1,2,3,5,6,7,4,2,3,5,6,7],
                         k=[2,3,0,6,7,4,1,3,0,6,7,4],
                         color='lightgray', opacity=0.05, showscale=False)
    
        fig = go.Figure(data=[mesh, trace_pores, trace_bio])
        fig.update_layout(scene=dict(aspectmode='cube', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                          margin=dict(l=0, r=0, t=25, b=0),
                          title=f"Micro cube: {cube_size_mm} mm — biochar {biochar_pct:.1f}% — porosity ~ {porosity:.3f}")
        camera = dict(eye=dict(x=0.8, y=0.8, z=0.8))
        fig.update_layout(scene_camera=camera)
        return fig, bio_pts, pore_pts, bio_r_mm, pore_r_mm
    
    # -------------------------
    # ---- Compute derived properties
    # -------------------------
    porosity = estimate_porosity(biochar_pct, wc_ratio, aggregate_pct)
    co2_uptake = estimate_co2_uptake_per_m3(biochar_pct, porosity)
    strength = estimate_strength(biochar_pct, porosity, wc_ratio)
    
    # Display computed numbers
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated porosity (vol frac)", f"{porosity:.3f}")
    col2.metric("Est. CO₂ uptake (kg / m³)", f"{co2_uptake:.1f}")
    col3.metric("Est. 28d compressive strength (MPa)", f"{strength:.1f}")
    
    # -------------------------
    # ---- Create / show figures
    # -------------------------
    if scale_choice == "Macro (panel)":
        fig = create_macro_figure(panel_w, panel_h, co2_uptake, biochar_pct)
        st.plotly_chart(fig, use_container_width=True, height=600)
    
    elif scale_choice == "Meso (cutout)":
        fig = create_meso_figure(micro_cube_mm, porosity, grid_n=18)
        st.plotly_chart(fig, use_container_width=True, height=700)
    
    elif scale_choice == "Micro (zoom)":
        fig, bio_pts, pore_pts, bio_r_mm, pore_r_mm = create_micro_figure(micro_cube_mm, biochar_pct, porosity, micro_particle_count)
        st.plotly_chart(fig, use_container_width=True, height=700)
    
        if export_csv:
            # prepare a small dataframe with particle data
            bio_df = pd.DataFrame(bio_pts, columns=['x','y','z'])
            bio_df['type'] = 'biochar'
            bio_df['radius_mm'] = bio_r_mm
            pore_df = pd.DataFrame(pore_pts, columns=['x','y','z'])
            pore_df['type'] = 'pore'
            pore_df['radius_mm'] = pore_r_mm
            df = pd.concat([bio_df, pore_df], ignore_index=True)
    
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(label='Download microstructure CSV', data=csv_bytes, file_name='microstructure_pointcloud.csv', mime='text/csv')
    
    elif scale_choice == "Play zoom animation":
        st.info("Playing quick zoom: Macro → Meso → Micro")
        f1 = create_macro_figure(panel_w, panel_h, co2_uptake, biochar_pct)
        st.plotly_chart(f1, use_container_width=True, height=500)
        time.sleep(0.6)
        f2 = create_meso_figure(micro_cube_mm, porosity, grid_n=14)
        st.plotly_chart(f2, use_container_width=True, height=500)
        time.sleep(0.6)
        f3, bio_pts, pore_pts, _, _ = create_micro_figure(micro_cube_mm, biochar_pct, porosity, min(1200, micro_particle_count))
        st.plotly_chart(f3, use_container_width=True, height=500)
    
    # -------------------------
    # ---- Assumptions & checks (explicit list)
    # -------------------------
    with st.expander("Assumptions & checks (open and review)"):
        st.markdown("""
        **Explicit assumptions used by the app (please verify / replace):**
    
        1. **Representative volumes**: Micro cube default is 10 mm; you should check whether this scale matches the features you want to represent (SEM/CT scales vary).
        2. **Concrete density**: 2400 kg/m^3 used to convert biochar % (by mass) into kg biochar per m^3. Verify for your mix. (ASSUMPTION)
        3. **Porosity model**: a linear, synthetic relation `base + alpha*biochar + beta*(w/c-0.4) - gamma*aggregate` is used. Replace with an experimental or literature correlation. (ASSUMPTION)
        4. **CO₂ adsorption capacity**: 0.25 kg CO₂ per kg biochar (fictional). Replace with BET / adsorption data for your biochar. (ASSUMPTION)
        5. **Particle shapes**: particles and pores are approximated as spheres for visualization. Real biochar is irregular — check morphology with SEM / CT. (ASSUMPTION)
        6. **Overlap & packing**: the micro generator does NOT perform collision/packing — particles can overlap. Use a packing algorithm for more realistic geometry. (ASSUMPTION)
        7. **Particle size distributions**: synthetic lognormal distributions are used with median sizes given in code; replace with measured PSDs. (ASSUMPTION)
        8. **Visualization simplifications**: Plotly uses marker points (not true 3D meshes) for speed. For photorealistic or high-fidelity rendering export to Blender or a mesh format. (ASSUMPTION)
        9. **Performance**: browser rendering slows down if you push particle counts > ~5000. Keep counts moderate while prototyping.
    
        **Suggested checks / next steps:**
        - Replace estimate_porosity(), estimate_co2_uptake_per_m3(), estimate_strength() with your own formulas or regressions.
        - If you have CT data: segment it and export particle centroids / meshes to import here or into Blender for high-fidelity visuals.
        - For realistic pore networks consider using PoreSpy (python) or packing libraries to generate non-overlapping particle geometries.
        - For validation: compare generated porosity with mercury intrusion porosimetry / BET / helium pycnometry results.
        """)
    
    # -------------------------
    # ---- End
    # -------------------------
    
    st.markdown("---")
    st.caption("This app is a prototype: replace placeholder relationships with literature/experimental models. Ask me to help tune any of the placeholder functions for your literature values.")

