import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

from simulation import create_cars
from kalman import create_filters
from collision import analyze_collision
from avoidance import find_best_manoeuvre
from digital_twin import create_digital_twin
from heatmap import generate_risk_heatmap
from reentry import simulate_reentry


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Space Collision Monitor",
    page_icon="🚀",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚀 Space Collision Detection & Avoidance System")

st.caption(
    "Software-in-the-Loop Prototype | "
    "Two-Object Digital Twin Simulation"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Simulation Controls")

simulation_speed = st.sidebar.slider(
    "Object A Speed",
    0.5,
    5.0,
    1.5,
    0.1
)

object_b_speed = st.sidebar.slider(
    "Object B Speed",
    0.5,
    5.0,
    1.2,
    0.1
)

noise_level = st.sidebar.slider(
    "Sensor Noise",
    0.0,
    1.0,
    0.3,
    0.05
)

run_simulation = st.sidebar.checkbox(
    "Run Live Simulation",
    value=False
)


# ============================================================
# INITIALIZE CARS
# ============================================================

if "initialized" not in st.session_state:

    car_a, car_b = create_cars()

    car_a.velocity[0] = simulation_speed
    car_b.velocity[0] = -object_b_speed

    filter_a, filter_b = create_filters()

    st.session_state.car_a = car_a
    st.session_state.car_b = car_b

    st.session_state.filter_a = filter_a
    st.session_state.filter_b = filter_b

    st.session_state.initialized = True


car_a = st.session_state.car_a
car_b = st.session_state.car_b

filter_a = st.session_state.filter_a
filter_b = st.session_state.filter_b


# Update velocities from sidebar

car_a.velocity[0] = simulation_speed
car_b.velocity[0] = -object_b_speed


# ============================================================
# SIMULATE SENSOR DATA
# ============================================================

measurement_a = car_a.get_noisy_measurement(
    noise_level
)

measurement_b = car_b.get_noisy_measurement(
    noise_level
)


# ============================================================
# KALMAN FILTER
# ============================================================

filtered_a = filter_a.update([
    measurement_a["x"],
    measurement_a["y"],
    measurement_a["z"]
])

filtered_b = filter_b.update([
    measurement_b["x"],
    measurement_b["y"],
    measurement_b["z"]
])


# ============================================================
# COLLISION ANALYSIS
# ============================================================

analysis = analyze_collision(
    filtered_a,
    car_a.velocity,
    filtered_b,
    car_b.velocity
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader("Live Collision Status")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Distance",
        f"{analysis['distance']:.2f} m"
    )

with c2:

    st.metric(
        "Relative Velocity",
        f"{analysis['relative_speed']:.2f} m/s"
    )

with c3:

    st.metric(
        "Time to CPA",
        f"{analysis['time_to_cpa']:.2f} s"
    )

with c4:

    st.metric(
        "Closest Approach",
        f"{analysis['minimum_distance']:.2f} m"
    )

with c5:

    st.metric(
        "Collision Probability",
        f"{analysis['probability']}%"
    )


# ============================================================
# RISK STATUS
# ============================================================

st.divider()

risk = analysis["risk"]

if risk == "CRITICAL":

    st.error("🚨 CRITICAL COLLISION RISK")

elif risk == "HIGH":

    st.warning("⚠ HIGH COLLISION RISK")

elif risk == "MEDIUM":

    st.warning("🟡 MEDIUM COLLISION RISK")

else:

    st.success("🟢 LOW COLLISION RISK")


# ============================================================
# OBJECT TELEMETRY
# ============================================================

st.subheader("📡 Object Telemetry")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🚗 Object A")

    st.write(
        f"X: `{filtered_a[0]:.2f} m`"
    )

    st.write(
        f"Y: `{filtered_a[1]:.2f} m`"
    )

    st.write(
        f"Z: `{filtered_a[2]:.2f} m`"
    )

    speed_a = np.linalg.norm(
        car_a.velocity
    )

    st.write(
        f"Speed: `{speed_a:.2f} m/s`"
    )

with col2:

    st.markdown("### 🚗 Object B")

    st.write(
        f"X: `{filtered_b[0]:.2f} m`"
    )

    st.write(
        f"Y: `{filtered_b[1]:.2f} m`"
    )

    st.write(
        f"Z: `{filtered_b[2]:.2f} m`"
    )

    speed_b = np.linalg.norm(
        car_b.velocity
    )

    st.write(
        f"Speed: `{speed_b:.2f} m/s`"
    )


# ============================================================
# KALMAN FILTER SECTION
# ============================================================

st.divider()

st.subheader("📈 Kalman Filter State Estimation")

k1, k2 = st.columns(2)

with k1:

    st.write("Object A")

    st.write(
        f"Raw X measurement: "
        f"**{measurement_a['x']:.2f} m**"
    )

    st.write(
        f"Filtered X estimate: "
        f"**{filtered_a[0]:.2f} m**"
    )

with k2:

    st.write("Object B")

    st.write(
        f"Raw X measurement: "
        f"**{measurement_b['x']:.2f} m**"
    )

    st.write(
        f"Filtered X estimate: "
        f"**{filtered_b[0]:.2f} m**"
    )


# ============================================================
# AVOIDANCE
# ============================================================

st.divider()

st.subheader("🛡 Collision Avoidance Planner")

best_manoeuvre, manoeuvres = find_best_manoeuvre(
    filtered_a,
    car_a.velocity,
    filtered_b,
    car_b.velocity
)

if risk in ["HIGH", "CRITICAL"]:

    st.error(
        "Collision risk detected — avoidance required."
    )

    st.success(
        f"Recommended manoeuvre: "
        f"**{best_manoeuvre['manoeuvre']}**"
    )

    st.write(
        f"Predicted minimum separation: "
        f"**{best_manoeuvre['minimum_distance']:.2f} m**"
    )

    st.write(
        f"Estimated energy cost: "
        f"**{best_manoeuvre['energy']:.2f} units**"
    )

else:

    st.success(
        "Current trajectories are within the safe region."
    )


# ============================================================
# MANOEUVRE COMPARISON
# ============================================================

st.subheader("Manoeuvre Comparison")

for manoeuvre in manoeuvres:

    st.write(
        f"**{manoeuvre['manoeuvre']}** — "
        f"Minimum separation: "
        f"{manoeuvre['minimum_distance']:.2f} m | "
        f"Energy: {manoeuvre['energy']:.2f}"
    )


# ============================================================
# DIGITAL TWIN
# ============================================================

st.divider()

st.subheader("🌐 3D Digital Twin")

twin = create_digital_twin(
    filtered_a,
    car_a.velocity,
    filtered_b,
    car_b.velocity
)

trajectory_a = twin["trajectory_a"]
trajectory_b = twin["trajectory_b"]

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=trajectory_a[0],
        y=trajectory_a[1],
        z=trajectory_a[2],
        mode="lines",
        name="Object A trajectory"
    )
)

fig.add_trace(
    go.Scatter3d(
        x=trajectory_b[0],
        y=trajectory_b[1],
        z=trajectory_b[2],
        mode="lines",
        name="Object B trajectory"
    )
)

fig.add_trace(
    go.Scatter3d(
        x=[filtered_a[0]],
        y=[filtered_a[1]],
        z=[filtered_a[2]],
        mode="markers",
        marker=dict(size=8),
        name="Object A"
    )
)

fig.add_trace(
    go.Scatter3d(
        x=[filtered_b[0]],
        y=[filtered_b[1]],
        z=[filtered_b[2]],
        mode="markers",
        marker=dict(size=8),
        name="Object B"
    )
)

fig.update_layout(
    height=650,
    scene=dict(
        xaxis_title="X Position",
        yaxis_title="Y Position",
        zaxis_title="Z Position"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HEATMAP
# ============================================================

st.divider()

st.subheader("🔥 Collision Risk Heatmap")

X, Y, risk_map = generate_risk_heatmap(
    filtered_a,
    filtered_b
)

heatmap_fig = go.Figure(
    data=go.Heatmap(
        x=X[0],
        y=Y[:, 0],
        z=risk_map
    )
)

heatmap_fig.update_layout(
    xaxis_title="X Position",
    yaxis_title="Y Position",
    height=500
)

st.plotly_chart(
    heatmap_fig,
    use_container_width=True
)


# ============================================================
# RE-ENTRY SIMULATION
# ============================================================

st.divider()

st.subheader("🛰 Re-entry Prediction Module")

time_data, altitude, reentry_time = simulate_reentry()

reentry_fig = go.Figure()

reentry_fig.add_trace(
    go.Scatter(
        x=time_data,
        y=altitude,
        mode="lines",
        name="Altitude"
    )
)

reentry_fig.update_layout(
    xaxis_title="Simulation Time",
    yaxis_title="Altitude (km)",
    height=400
)

st.plotly_chart(
    reentry_fig,
    use_container_width=True
)

if reentry_time is not None:

    st.info(
        f"Estimated re-entry threshold reached at "
        f"{reentry_time:.1f} simulation time units."
    )

else:

    st.info(
        "Object remains above the simulated "
        "re-entry threshold."
    )


# ============================================================
# SYSTEM ARCHITECTURE STATUS
# ============================================================

st.divider()

st.subheader("🧠 System Intelligence Modules")

modules = {
    "Virtual Telemetry": "ACTIVE",
    "Kalman State Estimation": "ACTIVE",
    "Collision Prediction": "ACTIVE",
    "Avoidance Planner": "ACTIVE",
    "3D Digital Twin": "ACTIVE",
    "Energy-Aware Avoidance": "ACTIVE",
    "Collision Heatmap": "ACTIVE",
    "Re-entry Simulation": "SIMULATION"
}

for module, status in modules.items():

    st.write(
        f"**{module}:** `{status}`"
    )


# ============================================================
# LIVE SIMULATION
# ============================================================

if run_simulation:

    time.sleep(0.1)

    car_a.update(0.1)
    car_b.update(0.1)

    st.rerun()