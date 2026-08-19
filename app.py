import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import urllib3

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from simulation import create_cars
from kalman import create_filters
from avoidance import find_best_manoeuvre
from energy import compare_manoeuvres
from heatmap import generate_risk_heatmap
from reentry import simulate_reentry


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Space Collision & Avoidance Monitor",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# DISABLE HTTPS WARNING FOR CELESTRAK
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛰️ Space Collision & Avoidance Monitor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Digital Twin + Kalman Filter + Collision Prediction + '
    'Risk Heatmap + Energy-Aware Avoidance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Controls")


live_mode = st.sidebar.toggle(
    "🔴 Live Simulation",
    value=True
)


simulation_speed = st.sidebar.slider(
    "Simulation Speed",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)


manual_simulation_time = st.sidebar.slider(
    "Manual Simulation Time (s)",
    min_value=0.0,
    max_value=30.0,
    value=5.0,
    step=0.5
)


prediction_duration = st.sidebar.slider(
    "Prediction Duration (s)",
    min_value=5,
    max_value=30,
    value=10
)


noise_level = st.sidebar.slider(
    "Sensor Noise",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1
)


safe_distance = st.sidebar.slider(
    "Safe Distance (m)",
    min_value=1,
    max_value=20,
    value=5
)


st.sidebar.markdown("---")


st.sidebar.info(
    """
    ### Prototype Mode

    🚗 Object A = Vehicle 1

    🚙 Object B = Vehicle 2

    The vehicles represent two
    satellites/space objects in the
    SIH prototype.
    """
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "simulation_start_time" not in st.session_state:

    st.session_state.simulation_start_time = time.time()


if "filter_a" not in st.session_state:

    st.session_state.filter_a, st.session_state.filter_b = (
        create_filters()
    )


if "car_a" not in st.session_state:

    st.session_state.car_a, st.session_state.car_b = (
        create_cars()
    )


if "last_heatmap_time" not in st.session_state:

    st.session_state.last_heatmap_time = 0.0


if "last_manoeuvre_time" not in st.session_state:

    st.session_state.last_manoeuvre_time = 0.0


if "last_manoeuvre_result" not in st.session_state:

    st.session_state.last_manoeuvre_result = None


if "last_energy_data" not in st.session_state:

    st.session_state.last_energy_data = None


if "current_state" not in st.session_state:

    st.session_state.current_state = {}


# ============================================================
# INITIAL OBJECT CONDITIONS
# ============================================================

INITIAL_POSITION_A = np.array(
    [0.0, 0.0, 2.0]
)


VELOCITY_A = np.array(
    [1.5, 0.1, 0.0]
)


INITIAL_POSITION_B = np.array(
    [30.0, 3.0, 2.0]
)


VELOCITY_B = np.array(
    [-1.2, -0.05, 0.0]
)


# ============================================================
# LIVE MONITORING FRAGMENT
#
# ONLY THIS SECTION RUNS EVERY 300 ms.
# THE REST OF THE PAGE DOES NOT REBUILD.
# ============================================================

@st.fragment(run_every="300ms")
def live_monitoring():

    # ========================================================
    # SIMULATION TIME
    # ========================================================

    if live_mode:

        elapsed_time = (
            time.time()
            -
            st.session_state.simulation_start_time
        )

        simulation_time = (
            elapsed_time
            *
            simulation_speed
        )

    else:

        simulation_time = manual_simulation_time


    # ========================================================
    # CURRENT POSITIONS
    # ========================================================

    position_a = (
        INITIAL_POSITION_A
        +
        VELOCITY_A
        *
        simulation_time
    )


    position_b = (
        INITIAL_POSITION_B
        +
        VELOCITY_B
        *
        simulation_time
    )


    # ========================================================
    # UPDATE CAR OBJECTS
    # ========================================================

    st.session_state.car_a.position = position_a
    st.session_state.car_a.velocity = VELOCITY_A

    st.session_state.car_b.position = position_b
    st.session_state.car_b.velocity = VELOCITY_B


    # ========================================================
    # SENSOR MEASUREMENTS
    # ========================================================

    measurement_a = (
        position_a
        +
        np.random.normal(
            0,
            noise_level,
            3
        )
    )


    measurement_b = (
        position_b
        +
        np.random.normal(
            0,
            noise_level,
            3
        )
    )


    # ========================================================
    # KALMAN FILTER
    #
    # IMPORTANT:
    # The filters are persistent.
    # They are NOT recreated every frame.
    # ========================================================

    filtered_a = (
        st.session_state.filter_a.update(
            measurement_a
        )
    )


    filtered_b = (
        st.session_state.filter_b.update(
            measurement_b
        )
    )


    # ========================================================
    # RELATIVE POSITION
    # ========================================================

    relative_position = (
        position_b
        -
        position_a
    )


    # ========================================================
    # CURRENT DISTANCE
    # ========================================================

    current_distance = np.linalg.norm(
        relative_position
    )


    # ========================================================
    # FILTERED DISTANCE
    # ========================================================

    filtered_distance = np.linalg.norm(
        filtered_b
        -
        filtered_a
    )


    # ========================================================
    # RELATIVE VELOCITY
    # ========================================================

    relative_velocity = (
        VELOCITY_B
        -
        VELOCITY_A
    )


    relative_speed = np.linalg.norm(
        relative_velocity
    )


    # ========================================================
    # TIME TO CLOSEST APPROACH
    # ========================================================

    velocity_squared = np.dot(
        relative_velocity,
        relative_velocity
    )


    if velocity_squared > 0:

        tca = (
            -
            np.dot(
                relative_position,
                relative_velocity
            )
            /
            velocity_squared
        )

    else:

        tca = 0.0


    tca = max(
        tca,
        0.0
    )


    # ========================================================
    # PREDICTED CLOSEST APPROACH
    # ========================================================

    future_a = (
        position_a
        +
        VELOCITY_A
        *
        tca
    )


    future_b = (
        position_b
        +
        VELOCITY_B
        *
        tca
    )


    minimum_distance = np.linalg.norm(
        future_b
        -
        future_a
    )


    # ========================================================
    # COLLISION RISK
    # ========================================================

    if minimum_distance < safe_distance:

        risk_level = "CRITICAL"
        risk_score = 95

    elif minimum_distance < safe_distance * 2:

        risk_level = "WARNING"
        risk_score = 65

    else:

        risk_level = "SAFE"
        risk_score = 15


    # ========================================================
    # SAVE CURRENT STATE
    #
    # Other parts of the application can use this state.
    # ========================================================

    st.session_state.current_state = {

        "position_a": position_a,
        "position_b": position_b,

        "velocity_a": VELOCITY_A,
        "velocity_b": VELOCITY_B,

        "measurement_a": measurement_a,
        "measurement_b": measurement_b,

        "filtered_a": filtered_a,
        "filtered_b": filtered_b,

        "current_distance": current_distance,
        "filtered_distance": filtered_distance,

        "relative_speed": relative_speed,

        "tca": tca,

        "future_a": future_a,
        "future_b": future_b,

        "minimum_distance": minimum_distance,

        "risk_level": risk_level,
        "risk_score": risk_score,

        "simulation_time": simulation_time
    }


    # ========================================================
    # LIVE HEADER
    # ========================================================

    st.subheader(
        "🔴 LIVE Collision Monitoring"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Simulation Time",
            f"{simulation_time:.1f} s"
        )


    with col2:

        st.metric(
            "Current Distance",
            f"{current_distance:.2f} m"
        )


    with col3:

        st.metric(
            "Closest Distance",
            f"{minimum_distance:.2f} m"
        )


    with col4:

        st.metric(
            "Relative Speed",
            f"{relative_speed:.2f} m/s"
        )


    with col5:

        st.metric(
            "TCA",
            f"{tca:.2f} s"
        )


    # ========================================================
    # RISK ALERT
    # ========================================================

    if risk_level == "CRITICAL":

        st.error(
            f"🚨 CRITICAL COLLISION RISK — {risk_score}%"
        )

    elif risk_level == "WARNING":

        st.warning(
            f"⚠️ COLLISION WARNING — {risk_score}%"
        )

    else:

        st.success(
            f"✅ COLLISION RISK LOW — {risk_score}%"
        )


    # ========================================================
    # 3D DIGITAL TWIN
    # ========================================================

    st.subheader(
        "🌐 Live 3D Digital Twin"
    )


    trajectory_time = np.linspace(
        0,
        prediction_duration,
        150
    )


    trajectory_a = (
        position_a[:, None]
        +
        VELOCITY_A[:, None]
        *
        trajectory_time
    )


    trajectory_b = (
        position_b[:, None]
        +
        VELOCITY_B[:, None]
        *
        trajectory_time
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig_3d = go.Figure()


    # ========================================================
    # OBJECT A TRAJECTORY
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=trajectory_a[0],
            y=trajectory_a[1],
            z=trajectory_a[2],
            mode="lines",
            name="Object A Trajectory",
            line=dict(
                width=5
            )
        )
    )


    # ========================================================
    # OBJECT B TRAJECTORY
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=trajectory_b[0],
            y=trajectory_b[1],
            z=trajectory_b[2],
            mode="lines",
            name="Object B Trajectory",
            line=dict(
                width=5
            )
        )
    )


    # ========================================================
    # OBJECT A
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[position_a[0]],
            y=[position_a[1]],
            z=[position_a[2]],
            mode="markers+text",
            text=["🚗 Object A"],
            textposition="top center",
            marker=dict(
                size=12
            ),
            name="Object A"
        )
    )


    # ========================================================
    # OBJECT B
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[position_b[0]],
            y=[position_b[1]],
            z=[position_b[2]],
            mode="markers+text",
            text=["🚙 Object B"],
            textposition="top center",
            marker=dict(
                size=12
            ),
            name="Object B"
        )
    )


    # ========================================================
    # CURRENT SEPARATION
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[
                position_a[0],
                position_b[0]
            ],
            y=[
                position_a[1],
                position_b[1]
            ],
            z=[
                position_a[2],
                position_b[2]
            ],
            mode="lines",
            name="Current Separation",
            line=dict(
                width=3,
                dash="dash"
            )
        )
    )


    # ========================================================
    # CLOSEST APPROACH A
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[future_a[0]],
            y=[future_a[1]],
            z=[future_a[2]],
            mode="markers",
            marker=dict(
                size=8,
                symbol="x"
            ),
            name="Closest Approach A"
        )
    )


    # ========================================================
    # CLOSEST APPROACH B
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[future_b[0]],
            y=[future_b[1]],
            z=[future_b[2]],
            mode="markers",
            marker=dict(
                size=8,
                symbol="x"
            ),
            name="Closest Approach B"
        )
    )


    # ========================================================
    # CLOSEST APPROACH LINE
    # ========================================================

    fig_3d.add_trace(
        go.Scatter3d(
            x=[
                future_a[0],
                future_b[0]
            ],
            y=[
                future_a[1],
                future_b[1]
            ],
            z=[
                future_a[2],
                future_b[2]
            ],
            mode="lines",
            name="Predicted Closest Separation",
            line=dict(
                width=3,
                dash="dot"
            )
        )
    )


    # ========================================================
    # 3D LAYOUT
    # ========================================================

    fig_3d.update_layout(

        title=(
            "Live 3D Digital Twin — "
            f"t = {simulation_time:.1f} s"
        ),

        height=700,

        template="plotly_dark",

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50
        ),

        scene=dict(

            xaxis=dict(
                title="X Position (m)",
                showgrid=True,
                zeroline=True
            ),

            yaxis=dict(
                title="Y Position (m)",
                showgrid=True,
                zeroline=True
            ),

            zaxis=dict(
                title="Z Position (m)",
                showgrid=True,
                zeroline=True
            ),

            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.2
                )
            )
        )
    )


    st.plotly_chart(
        fig_3d,
        use_container_width=True,
        key="live_3d_plot"
    )


    # ========================================================
    # LIVE KALMAN TRACKING
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧮 Live Kalman Filter Tracking"
    )


    kalman_df = pd.DataFrame({

        "Object": [
            "Object A",
            "Object B"
        ],

        "True X": [
            position_a[0],
            position_b[0]
        ],

        "True Y": [
            position_a[1],
            position_b[1]
        ],

        "Measured X": [
            measurement_a[0],
            measurement_b[0]
        ],

        "Measured Y": [
            measurement_a[1],
            measurement_b[1]
        ],

        "Kalman X": [
            filtered_a[0],
            filtered_b[0]
        ],

        "Kalman Y": [
            filtered_a[1],
            filtered_b[1]
        ]
    })


    st.dataframe(
        kalman_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # LIVE OBJECT STATE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📍 Live Object State"
    )


    state_df = pd.DataFrame({

        "Parameter": [

            "Object A X",
            "Object A Y",
            "Object A Z",

            "Object A Vx",
            "Object A Vy",
            "Object A Vz",

            "Object B X",
            "Object B Y",
            "Object B Z",

            "Object B Vx",
            "Object B Vy",
            "Object B Vz"
        ],

        "Value": [

            position_a[0],
            position_a[1],
            position_a[2],

            VELOCITY_A[0],
            VELOCITY_A[1],
            VELOCITY_A[2],

            position_b[0],
            position_b[1],
            position_b[2],

            VELOCITY_B[0],
            VELOCITY_B[1],
            VELOCITY_B[2]
        ]
    })


    st.dataframe(
        state_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RISK HEATMAP
    #
    # Recalculate approximately once per second.
    # ========================================================

    current_real_time = time.time()


    if (
        current_real_time
        -
        st.session_state.last_heatmap_time
        >= 1.0
    ):

        X, Y, risk = generate_risk_heatmap(
            position_a,
            position_b
        )


        heatmap_fig = go.Figure(
            data=go.Heatmap(
                x=X[0],
                y=Y[:, 0],
                z=risk,
                colorscale="Turbo",
                colorbar=dict(
                    title="Risk"
                )
            )
        )


        heatmap_fig.add_trace(
            go.Scatter(
                x=[position_a[0]],
                y=[position_a[1]],
                mode="markers+text",
                text=["Object A"],
                textposition="top center",
                marker=dict(
                    size=15
                ),
                name="Object A"
            )
        )


        heatmap_fig.add_trace(
            go.Scatter(
                x=[position_b[0]],
                y=[position_b[1]],
                mode="markers+text",
                text=["Object B"],
                textposition="top center",
                marker=dict(
                    size=15
                ),
                name="Object B"
            )
        )


        heatmap_fig.update_layout(
            title="Live Spatial Collision Risk",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            height=600
        )


        st.session_state.heatmap_fig = heatmap_fig

        st.session_state.last_heatmap_time = (
            current_real_time
        )


    # Display latest heatmap
    if "heatmap_fig" in st.session_state:

        st.markdown("---")

        st.subheader(
            "🌡️ Live Collision Risk Heatmap"
        )

        st.plotly_chart(
            st.session_state.heatmap_fig,
            use_container_width=True,
            key="live_heatmap"
        )


    # ========================================================
    # MANOEUVRE ANALYSIS
    #
    # Only calculate when WARNING/CRITICAL.
    # Recalculate at most once per second.
    # ========================================================

    if risk_level in ["WARNING", "CRITICAL"]:

        if (
            current_real_time
            -
            st.session_state.last_manoeuvre_time
            >= 1.0
        ):

            best_manoeuvre, manoeuvre_results = (
                find_best_manoeuvre(
                    position_a,
                    VELOCITY_A,
                    position_b,
                    VELOCITY_B
                )
            )


            st.session_state.last_manoeuvre_result = (
                best_manoeuvre,
                manoeuvre_results
            )


            current_speed = np.linalg.norm(
                VELOCITY_A
            )


            st.session_state.last_energy_data = (
                compare_manoeuvres(
                    current_speed,
                    duration=5
                )
            )


            st.session_state.last_manoeuvre_time = (
                current_real_time
            )


    # ========================================================
    # MANOEUVRE DISPLAY
    # ========================================================

    if st.session_state.last_manoeuvre_result:

        best_manoeuvre, manoeuvre_results = (
            st.session_state.last_manoeuvre_result
        )


        st.markdown("---")

        st.subheader(
            "🧭 Collision Avoidance Manoeuvres"
        )


        manoeuvre_df = pd.DataFrame(
            manoeuvre_results
        )


        manoeuvre_df.columns = [
            "Manoeuvre",
            "Minimum Distance (m)",
            "Energy Cost"
        ]


        st.dataframe(
            manoeuvre_df,
            use_container_width=True,
            hide_index=True
        )


        st.success(
            f"""
            🏆 **Recommended Manoeuvre:**
            {best_manoeuvre["manoeuvre"]}

            Minimum predicted separation:
            **{best_manoeuvre["minimum_distance"]:.2f} m**

            Estimated energy cost:
            **{best_manoeuvre["energy"]:.2f} units**
            """
        )


        # ====================================================
        # ENERGY ANALYSIS
        # ====================================================

        if st.session_state.last_energy_data:

            st.subheader(
                "⛽ Energy / Fuel Analysis"
            )


            energy_df = pd.DataFrame(
                list(
                    st.session_state.last_energy_data.items()
                ),
                columns=[
                    "Strategy",
                    "Energy Cost"
                ]
            )


            st.bar_chart(
                energy_df.set_index(
                    "Strategy"
                )
            )


# ============================================================
# START LIVE MONITORING
# ============================================================

live_monitoring()


# ============================================================
# RE-ENTRY SIMULATION
#
# This does NOT need to update every 300 ms.
# ============================================================

st.markdown("---")

st.subheader(
    "🌍 Estimated Re-entry Prediction"
)


reentry_time_array, altitude, reentry_time = (
    simulate_reentry()
)


reentry_fig = go.Figure()


reentry_fig.add_trace(
    go.Scatter(
        x=reentry_time_array,
        y=altitude,
        mode="lines",
        name="Altitude"
    )
)


reentry_fig.add_hline(
    y=100,
    line_dash="dash",
    annotation_text="Re-entry Threshold"
)


reentry_fig.update_layout(
    title="Predicted Altitude Decay",
    xaxis_title="Time",
    yaxis_title="Altitude (km)",
    height=500
)


st.plotly_chart(
    reentry_fig,
    use_container_width=True
)


if reentry_time is not None:

    st.info(
        f"Estimated re-entry threshold reached at "
        f"**{reentry_time:.2f} time units**."
    )

else:

    st.success(
        "No re-entry predicted during the simulation window."
    )


# ============================================================
# CELESTRAK DATA
# ============================================================

st.markdown("---")

st.subheader(
    "🛰️ CelesTrak Satellite Data"
)


CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=last-30-days&FORMAT=json"
)


@st.cache_data(ttl=3600)
def load_celestrak():

    try:

        response = requests.get(
            CELESTRAK_URL,
            timeout=15,
            verify=False
        )

        response.raise_for_status()

        return response.json(), None

    except Exception as e:

        return None, str(e)


with st.spinner(
    "Loading CelesTrak data..."
):

    satellites, error = load_celestrak()


if satellites:

    st.success(
        f"Loaded {len(satellites)} objects from CelesTrak."
    )


    satellite_names = [

        obj.get(
            "OBJECT_NAME",
            "Unknown"
        )

        for obj in satellites
    ]


    selected_name = st.selectbox(
        "Select satellite/object",
        satellite_names
    )


    selected_object = next(
        (
            obj
            for obj in satellites
            if obj.get("OBJECT_NAME")
            == selected_name
        ),
        None
    )


    if selected_object:

        c1, c2, c3 = st.columns(3)


        with c1:

            st.write(
                "**Object Name:**",
                selected_object.get(
                    "OBJECT_NAME",
                    "N/A"
                )
            )


        with c2:

            st.write(
                "**NORAD ID:**",
                selected_object.get(
                    "NORAD_CAT_ID",
                    "N/A"
                )
            )


        with c3:

            st.write(
                "**Object Type:**",
                selected_object.get(
                    "OBJECT_TYPE",
                    "N/A"
                )
            )


else:

    st.warning(
        "CelesTrak data could not be loaded."
    )


    if error:

        st.caption(
            f"Error: {error}"
        )


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.markdown("---")

st.subheader(
    "🧠 Complete Collision Avoidance Pipeline"
)


st.code(
    """
        CELESTRAK / SENSOR DATA
                  │
                  ▼
          DATA PREPROCESSING
                  │
                  ▼
          NOISY MEASUREMENTS
                  │
                  ▼
           KALMAN FILTER
                  │
                  ▼
        POSITION + VELOCITY
                  │
                  ▼
        DIGITAL TWIN MODEL
                  │
                  ▼
       LIVE 3D VISUALIZATION
                  │
                  ▼
       TRAJECTORY PREDICTION
                  │
                  ▼
       CLOSEST APPROACH (TCA)
                  │
                  ▼
        COLLISION RISK MODEL
                  │
          ┌───────┴────────┐
          ▼                ▼
       SAFE/WARN       CRITICAL
          │                │
          │                ▼
          │        MANOEUVRE SEARCH
          │                │
          │                ▼
          │        ENERGY ANALYSIS
          │                │
          │                ▼
          │       BEST MANOEUVRE
          │
          ▼
       DASHBOARD
          │
          ├── LIVE 3D DIGITAL TWIN
          ├── LIVE DISTANCE
          ├── LIVE TCA
          ├── LIVE COLLISION RISK
          ├── LIVE KALMAN TRACKING
          ├── LIVE OBJECT STATE
          ├── LIVE RISK HEATMAP
          ├── MANOEUVRE ANALYSIS
          ├── ENERGY ANALYSIS
          ├── RE-ENTRY PREDICTION
          └── CELESTRAK DATA
    """,
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH Prototype | Space Debris Detection & Collision Avoidance"
)